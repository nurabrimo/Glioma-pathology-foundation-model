"""End-to-end fine-tuning of the tile encoder + GMA aggregator, EAGLE recipe (Campanella et al., Nat Med 2025;
github.com/chadvanderbilt/EAGLE training/train.py with train.slurm arguments), on any number of GPUs.

Recipe kept: 1 slide per optimizer step; K = 2,208 tiles per slide (23 encoder ranks x k_per_gpu 96); ViT-g encoder;
GMA(ndim=1536, dropout=True, n_classes=2); CrossEntropyLoss([1-pos_weight, pos_weight]), pos_weight 0.7; AdamW on both
models (no weight decay on bias / 1-d parameters), lr linear warm-up 2 epochs 0 -> 1e-6 then cosine to 1e-8, weight
decay cosine 0.04 -> 0.4, 20 epochs, drop 0.5, fp16 autocast on the encoder with GradScaler, flips / 90-degree
rotations / Gaussian blur, reseeding with 1634 + epoch.

Difference in mechanics only: EAGLE holds the graph of 96 tiles on each of 23 GPUs; here the same gradient is obtained
with a two-pass scheme (core.py) so that K tiles fit on 1..N GPUs. Launch with torchrun for N > 1.
"""
import argparse
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import data as D
import eagle_utils as utils
import encoder as E
import modules

p = argparse.ArgumentParser()
# I/O
p.add_argument("--outdir", required=True)
p.add_argument("--slide_data", default="/work/nb295/eagle_glioma/slide_data.csv")
p.add_argument("--tiles", nargs="+", default=["/work/nb295/eagle_glioma/tiles"], help="parquet files / directories")
p.add_argument("--train_splits", nargs="+", default=["train"])
p.add_argument("--target", default="IDH")
p.add_argument("--backend", default="auto", choices=["auto", "cucim", "openslide"])
# model
p.add_argument("--init", default="gigapath", help="gigapath | eagle_lung | random | path to a tile checkpoint")
p.add_argument("--tilesize", type=int, default=224)
p.add_argument("--k", type=int, default=2208, help="tiles per slide per step (EAGLE: 23 x 96)")
p.add_argument("--drop", type=float, default=0.5)
p.add_argument("--pos_weight", type=float, default=0.7)
# optimisation (EAGLE train.slurm values)
p.add_argument("--optimizer", default="adamw", choices=["sgd", "adam", "adamw"])
p.add_argument("--momentum", type=float, default=0.9)
p.add_argument("--lr", type=float, default=1e-6)
p.add_argument("--lr_end", type=float, default=1e-8)
p.add_argument("--warmup_epochs", type=int, default=2)
p.add_argument("--weight_decay", type=float, default=0.04)
p.add_argument("--weight_decay_end", type=float, default=0.4)
p.add_argument("--use_amp", type=int, default=1, choices=[0, 1])
p.add_argument("--nepochs", type=int, default=20)
p.add_argument("--save_freq", type=int, default=2)
p.add_argument("--seed", type=int, default=1634)
# mechanics
p.add_argument("--chunk", type=int, default=0, help="tiles per forward/backward chunk in pass 2 (memory only, no effect on the gradient); 0 = from GPU memory")
p.add_argument("--load_bs", type=int, default=96, help="tiles per loader batch = pass-1 chunk")
p.add_argument("--workers", type=int, default=10)
p.add_argument("--prefetch", type=int, default=4)
p.add_argument("--ckpt_steps", type=int, default=0, help="save resumable checkpoint every N steps (0 = off)")
p.add_argument("--ckpt_minutes", type=float, default=30, help="... or every M minutes (0 = off)")
p.add_argument("--max_steps", type=int, default=0, help="stop after N steps in this invocation (tests)")
p.add_argument("--freeze_encoder", type=int, default=0)

STOP = {"flag": False}


def make_opt(name, groups, momentum):
    if name == "sgd":
        return optim.SGD(groups, lr=0., momentum=momentum, dampening=0, nesterov=True)
    if name == "adam":
        return optim.Adam(groups)
    return optim.AdamW(groups)


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = p.parse_args()
    amp = bool(args.use_amp)
    W = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if W > 1:
        dist.init_process_group(backend=os.environ.get("EG_DIST_BACKEND", "nccl"))  # gloo: test ranks sharing one GPU
    local = local % torch.cuda.device_count()
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main_rank = rank == 0
    os.makedirs(args.outdir, exist_ok=True)
    if args.chunk <= 0:
        # measured (ViT-g, fp16 autocast): ~19.5 GB fixed once AdamW state exists (weights 4.5 + grads 4.5 + Adam 9 + tiles)
        # + ~0.6 GB per tile of pass-2 activations; keep 3 GB headroom. 48 GB -> 36, 32 GB -> 12, 80 GB -> 92.
        tot = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30
        args.chunk = int(min(96, (tot - 24.0) / 0.6))
        if args.chunk < 8:
            raise SystemExit(f"GPU with {tot:.0f} GB is too small for end-to-end ViT-g fine-tuning (need >= 32 GB)")

    def log(*a):
        if main_rank:
            print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)

    signal.signal(signal.SIGTERM, lambda *_: STOP.update(flag=True))
    signal.signal(signal.SIGUSR1, lambda *_: STOP.update(flag=True))

    # data
    dfs = D.load_slides(args.slide_data, target=args.target, splits=args.train_splits)
    tiles = D.load_tiles(args.tiles, dfs.slide.astype(str))
    have = dfs.slide.astype(str).isin(tiles.keys())
    if (~have).any():
        log(f"WARNING: {int((~have).sum())} training slides have no tiles and are dropped: {dfs.slide[~have].tolist()[:5]}")
        dfs = dfs[have].reset_index(drop=True)
    dset = D.TrainTiles(dfs, tiles, args.k, tilesize=args.tilesize, drop=args.drop, seed=args.seed, backend=args.backend)
    nsteps = dset.nslides
    log(f"world={W} train slides={len(dfs)} (pos {int(dfs.target.sum())}) steps/epoch={nsteps} K={args.k} "
        f"per-rank tiles={args.k // W} chunk={args.chunk} amp={amp}")

    # models / optimisers
    enc = E.get_encoder(args.init, verbose=main_rank).to(dev)
    enc.train()
    if args.freeze_encoder:
        for q in enc.parameters():
            q.requires_grad_(False)
    slide_model = modules.GMA(ndim=E.NDIM, dropout=True, n_classes=2).to(dev)  # only rank 0's copy is used
    if args.pos_weight == 0.5:
        criterion = nn.CrossEntropyLoss().to(dev)
    else:
        criterion = nn.CrossEntropyLoss(torch.Tensor([1 - args.pos_weight, args.pos_weight])).to(dev)
    tile_opt = None if args.freeze_encoder else make_opt(args.optimizer, utils.get_params_groups(enc), args.momentum)
    slide_opt = make_opt(args.optimizer, utils.get_params_groups(slide_model), args.momentum)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # resume
    start_epoch, start_step, running_loss = 1, 0, 0.0
    last = os.path.join(args.outdir, "checkpoint_last.pth")
    if os.path.isfile(last):
        ck = E.safe_load(last, mmap=True)
        enc.load_state_dict(ck["tile_model"])
        slide_model.load_state_dict(ck["slide_model"])
        slide_opt.load_state_dict(ck["slide_optimizer"])
        if tile_opt is not None:
            tile_opt.load_state_dict(ck["tile_optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, start_step, running_loss = int(ck["epoch"]), int(ck["step"]), float(ck["running_loss"])
        del ck
        log(f"resumed from {last}: epoch {start_epoch} step {start_step}")

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.nepochs, nsteps, warmup_epochs=args.warmup_epochs)
    wd_schedule = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.nepochs, nsteps)

    outname = os.path.join(args.outdir, "convergence.csv")
    steplog = os.path.join(args.outdir, "steps.csv")
    if main_rank and not os.path.exists(outname):
        open(outname, "w").write("epoch,loss\n")
    if main_rank and not os.path.exists(steplog):
        open(steplog, "w").write("epoch,step,slide,target,loss,lr,t_step,t_wait,t_pass1,t_agg,t_pass2,t_sync_opt,peak_gb\n")

    def checkpoint(epoch, step, rl):
        if not main_rank:
            return
        t = time.time()
        save_atomic({"epoch": epoch, "step": step, "running_loss": rl,
                     "tile_model": enc.state_dict(),
                     "tile_optimizer": tile_opt.state_dict() if tile_opt is not None else {},
                     "scaler": scaler.state_dict(),
                     "slide_model": slide_model.state_dict(), "slide_optimizer": slide_opt.state_dict(),
                     "args": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in vars(args).items()}}, last)
        log(f"checkpoint epoch {epoch} step {step} saved in {time.time() - t:.0f}s")

    done_steps, t_ck = 0, time.time()
    per_rank = args.k // W
    for epoch in range(start_epoch, args.nepochs + 1):
        dset.makeData(epoch)
        s0 = start_step if epoch == start_epoch else 0
        if s0 == 0:
            running_loss = 0.0
        sampler = D.StepBatchSampler(nsteps, args.k, rank, W, args.load_bs, start_step=s0)
        loader = torch.utils.data.DataLoader(dset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                                             prefetch_factor=args.prefetch if args.workers > 0 else None)
        it_loader = iter(loader)
        for step in range(s0, nsteps):
            t0 = time.time()
            it = nsteps * (epoch - 1) + step
            for opt in (tile_opt, slide_opt):
                if opt is None:
                    continue
                for j, g in enumerate(opt.param_groups):
                    g["lr"] = lr_schedule[it]
                    if j == 0:
                        g["weight_decay"] = wd_schedule[it]
            torch.cuda.reset_peak_memory_stats(dev)
            # pass 1 (streamed with loading)
            imgs, feats, t_wait = [], [], 0.0
            for _ in range(sampler.per_step):
                tw = time.time()
                x = next(it_loader)
                t_wait += time.time() - tw
                x = x.to(dev, non_blocking=True)
                imgs.append(x)
                feats.append(core.encode_nograd(enc, x, amp))
            imgs, feats = torch.cat(imgs), torch.cat(feats)
            assert imgs.shape[0] == per_rank
            torch.cuda.synchronize(dev)
            t1 = time.time()
            # aggregator + pass 2
            label = torch.LongTensor([dset.get_target(step)]).to(dev)
            if tile_opt is not None:
                tile_opt.zero_grad(set_to_none=True)
            allf = core.gather_features(feats)
            loss, grads_all = None, None
            if main_rank:
                loss, grads_all = core.aggregate_gathered(slide_model, criterion, allf, label, slide_opt,
                                                          seed=(args.seed * 100003 + it) % (2 ** 31))
            torch.cuda.synchronize(dev)
            t2 = time.time()
            if tile_opt is not None:
                grads = core.share_grads(grads_all, tuple(allf.shape), dev, feats.dtype)
                core.encoder_backward(enc, imgs, grads, args.chunk, amp, scaler)
                torch.cuda.synchronize(dev)
                t3 = time.time()
                core.sync_grads(enc)
                scaler.step(tile_opt)
                scaler.update()
            else:
                t3 = time.time()
            torch.cuda.synchronize(dev)
            t4 = time.time()
            del imgs, feats, allf
            if main_rank:
                running_loss += loss
                peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
                open(steplog, "a").write(f"{epoch},{step},{dset.get_slide(step)},{label.item()},{loss:.5f},{lr_schedule[it]:.3e},"
                                         f"{t4 - t0:.2f},{t_wait:.2f},{t1 - t0:.2f},{t2 - t1:.2f},{t3 - t2:.2f},{t4 - t3:.2f},{peak:.1f}\n")
                log(f"{epoch}/{args.nepochs} - [{step + 1}/{nsteps}] - time: {t4 - t0:.1f} (wait {t_wait:.1f} p1 {t1 - t0:.1f} "
                    f"agg {t2 - t1:.2f} p2 {t3 - t2:.1f} sync+opt {t4 - t3:.1f}) - peak {peak:.1f} GB - loss: {loss:.3f}")
            done_steps += 1
            stop = bool(STOP["flag"] or (args.max_steps and done_steps >= args.max_steps))
            if W > 1:  # rank 0 decides, so that all ranks stop at the same step
                fl = torch.tensor([int(stop)], device=dev)
                dist.broadcast(fl, src=0)
                stop = bool(fl.item())
                STOP["flag"] = STOP["flag"] or stop
            due = (args.ckpt_steps and done_steps % args.ckpt_steps == 0) or \
                  (args.ckpt_minutes and time.time() - t_ck > 60 * args.ckpt_minutes)
            if step + 1 < nsteps and (stop or due):
                checkpoint(epoch, step + 1, running_loss)
                t_ck = time.time()
            if stop and step + 1 < nsteps:
                log("stopping (signal or max_steps); resume by relaunching with the same --outdir")
                del it_loader
                return
        del it_loader
        # end of epoch
        if main_rank:
            open(outname, "a").write(f"{epoch},{running_loss / dset.nslides}\n")
            if epoch % args.save_freq == 0 or epoch == args.nepochs:
                save_atomic({"epoch": epoch + 1, "tile_model": enc.state_dict()},
                            os.path.join(args.outdir, f"checkpoint_tile_{epoch:03}.pth"))
                save_atomic({"epoch": epoch + 1, "slide_model": slide_model.state_dict()},
                            os.path.join(args.outdir, f"checkpoint_slide_{epoch:03}.pth"))
        checkpoint(epoch + 1, 0, 0.0)
        t_ck = time.time()
        if STOP["flag"] or (args.max_steps and done_steps >= args.max_steps):
            return
    if main_rank:
        open(os.path.join(args.outdir, "DONE"), "w").write("done\n")
    log("TRAINING_DONE")


if __name__ == "__main__":
    main()
