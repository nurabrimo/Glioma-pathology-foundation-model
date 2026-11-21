"""Unit test: the two-pass chunked gradient (core.train_step) equals the direct full-graph gradient,
on one process and when the tiles are sharded over 2 processes (gloo, CPU). Tiny ViT + EAGLE GMA, float64."""
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import modules

K, NDIM, SEED = 48, 192, 7


def build():
    import timm
    torch.manual_seed(0)
    enc = timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=32, patch_size=8, num_classes=0,
                            init_values=1e-5).double()
    slide = modules.GMA(ndim=NDIM, dropout=True, n_classes=2).double()
    crit = nn.CrossEntropyLoss(torch.tensor([0.3, 0.7], dtype=torch.float64))
    g = torch.Generator().manual_seed(1)
    x = torch.randn(K, 3, 32, 32, generator=g, dtype=torch.float64)
    y = torch.LongTensor([1])
    enc.train(); slide.train()
    return enc, slide, crit, x, y


def grads_of(m):
    return torch.cat([p.grad.reshape(-1) for p in m.parameters()])


def direct():
    enc, slide, crit, x, y = build()
    f = enc(x)
    torch.manual_seed(SEED)
    _, _, out = slide(f)
    loss = crit(out, y)
    loss.backward()
    return loss.item(), grads_of(enc), grads_of(slide)


def worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(2)
    enc, slide, crit, x, y = build()
    xs = x[rank::world]  # same sharding as StepBatchSampler
    loss = core.train_step(enc, slide, crit, xs, y, chunk=5, amp=False, seed=SEED)
    if rank == 0:
        torch.save({"loss": loss, "enc": grads_of(enc), "slide": grads_of(slide)}, out)
    dist.destroy_process_group()


def main():
    torch.set_num_threads(4)
    l0, ge0, gs0 = direct()
    print(f"direct: loss {l0:.10f} |g_enc| {ge0.norm():.4e} max|g_enc| {ge0.abs().max():.4e} |g_agg| {gs0.norm():.4e}")
    ok = True
    for chunk in (K, 7):
        enc, slide, crit, x, y = build()
        l1 = core.train_step(enc, slide, crit, x, y, chunk=chunk, amp=False, seed=SEED)
        d = (grads_of(enc) - ge0).abs().max().item()
        ds = (grads_of(slide) - gs0).abs().max().item()
        print(f"two-pass 1 proc chunk={chunk}: loss diff {abs(l1 - l0):.2e} max|d g_enc| {d:.3e} "
              f"(rel {d / ge0.abs().max().item():.2e}) max|d g_agg| {ds:.3e}")
        ok &= d < 1e-9 * max(1.0, ge0.abs().max().item()) and ds < 1e-9
    # note: with W ranks the tile order seen by the aggregator differs (rank-major); the gradient is permutation invariant
    for world in (2, 3):
        out = os.path.join(tempfile.mkdtemp(), "g.pt")
        port = 29500 + world + os.getpid() % 1000
        ctx = mp.get_context("spawn")
        ps = [ctx.Process(target=worker, args=(r, world, port, out)) for r in range(world)]
        [q.start() for q in ps]; [q.join() for q in ps]
        assert all(q.exitcode == 0 for q in ps), "worker failed"
        r = torch.load(out)
        d = (r["enc"] - ge0).abs().max().item()
        ds = (r["slide"] - gs0).abs().max().item()
        print(f"two-pass {world} procs chunk=5: loss diff {abs(r['loss'] - l0):.2e} max|d g_enc| {d:.3e} "
              f"(rel {d / ge0.abs().max().item():.2e}) max|d g_agg| {ds:.3e}")
        ok &= d < 1e-9 * max(1.0, ge0.abs().max().item())
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
