"""Two-pass, chunked, optionally multi-process training step that reproduces the gradient of
EAGLE's train.py (encoder ranks -> features -> aggregator rank -> dL/dfeatures -> pseudo-loss backward).

EAGLE: each of 23 encoder ranks keeps the graph of 96 tiles, the aggregator rank returns dL/dfeatures, each rank
backpropagates (features * grads).sum() * 23 and DDP averages over the 23 ranks  ==  d loss / d theta summed over
all 2,208 tiles (the plain chain rule). Here the same gradient is accumulated chunk by chunk.
"""
import torch
import torch.distributed as dist


def world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


@torch.no_grad()
def encode_nograd(enc, x, amp):
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
        f = enc(x)
    return f.float() if f.dtype in (torch.float16, torch.bfloat16) else f


def aggregator_step(slide_model, criterion, feats, label, slide_opt=None, seed=None):
    """Forward/backward of the aggregator on detached features. Returns loss, dL/dfeatures, logits."""
    if seed is not None:
        torch.manual_seed(seed)  # dropout mask reproducible across resume
    feats = feats.detach().clone().requires_grad_()
    if slide_opt is not None:
        slide_opt.zero_grad()
    _, _, output = slide_model(feats)
    loss = criterion(output, label)
    loss.backward()
    grads = feats.grad.detach()
    if slide_opt is not None:
        slide_opt.step()
    return loss.item(), grads, output.detach()


def encoder_backward(enc, imgs, grads, chunk, amp, scaler=None):
    """Accumulate d(sum(features*grads))/d(theta) over chunks (pass 2)."""
    n = imgs.shape[0]
    for s in range(0, n, chunk):
        x = imgs[s:s + chunk]
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
            f = enc(x)
        f = f.to(grads.dtype)
        pl = (f * grads[s:s + chunk]).sum()
        if scaler is not None:
            scaler.scale(pl).backward()
        else:
            pl.backward()


def aggregate_gathered(slide_model, criterion, allf, label, slide_opt=None, seed=None):
    """Rank 0: allf (W, k, d) with rank r holding tiles r, r+W, ... -> restore the original tile order, run the
    aggregator, return loss and dL/dfeatures as (W, k, d)."""
    W, k, d = allf.shape
    loss, g, _ = aggregator_step(slide_model, criterion, allf.permute(1, 0, 2).reshape(W * k, d), label, slide_opt, seed)
    return loss, g.reshape(k, W, d).permute(1, 0, 2).contiguous()


def gather_features(feats):
    """All ranks call. Returns (W, k, d) on every rank (only rank 0 uses it)."""
    rank, W = world()
    if W == 1:
        return feats.unsqueeze(0)
    lst = [torch.empty_like(feats) for _ in range(W)]
    dist.all_gather(lst, feats.contiguous())
    return torch.stack(lst)


def share_grads(grads_all, shape, device, dtype=torch.float32):
    """Rank 0 passes (W, k, d); other ranks pass None. Returns this rank's (k, d)."""
    rank, W = world()
    if W == 1:
        return grads_all[0]
    buf = grads_all.contiguous() if rank == 0 else torch.empty(shape, device=device, dtype=dtype)
    dist.broadcast(buf, src=0)
    return buf[rank]


def sync_grads(model):
    """Sum encoder gradients over ranks so every rank holds the full-slide gradient."""
    rank, W = world()
    if W == 1:
        return
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


def train_step(enc, slide_model, criterion, imgs, label, chunk, amp, scaler=None,
               slide_opt=None, seed=None, feats=None, enc_chunk=None):
    """One slide. `imgs`: this rank's share of the K tiles (already on device).
    Leaves the summed gradient in enc.parameters() (.grad, scaled if scaler); the caller steps the encoder optimizer.
    Returns loss (rank 0, else None)."""
    rank, W = world()
    if feats is None:  # pass 1
        ec = enc_chunk or chunk
        feats = torch.cat([encode_nograd(enc, imgs[s:s + ec], amp) for s in range(0, imgs.shape[0], ec)])
    allf = gather_features(feats)  # (W, k, d)
    loss, grads_all = None, None
    if rank == 0:
        loss, grads_all = aggregate_gathered(slide_model, criterion, allf, label, slide_opt, seed)
    grads = share_grads(grads_all, tuple(allf.shape), feats.device, feats.dtype)
    encoder_backward(enc, imgs, grads, chunk, amp, scaler)  # pass 2
    sync_grads(enc)
    return loss
