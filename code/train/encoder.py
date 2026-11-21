"""Tile encoder (Prov-GigaPath ViT-g/16, 1,536-d) construction and checkpoint loading.

init choices
  gigapath   : timm "hf_hub:prov-gigapath/prov-gigapath" pretrained (the authors' initialisation; gated repo)
  eagle_lung : identical architecture built from an explicit timm config (no access to the gated repo needed),
               weights from MCCPBR/EAGLE tile_model.pth (GigaPath fine-tuned for lung EGFR)
  random     : identical architecture, random weights (tests only)
"""
import os
import numpy as np
import torch
import timm

NDIM = 1536


def build_arch():
    # Same config as prov-gigapath/prov-gigapath config.json:
    # vit_giant_patch14_dinov2, img_size 224, patch_size 16, embed 1536, depth 40, heads 24,
    # init_values 1e-5, mlp_ratio 5.33334 (SwiGLUPacked), num_classes 0
    return timm.create_model(
        "vit_giant_patch14_dinov2", pretrained=False, img_size=224, patch_size=16,
        in_chans=3, embed_dim=1536, depth=40, num_heads=24, init_values=1e-5,
        mlp_ratio=5.33334, num_classes=0)


def _safe_globals():
    import numpy._core.multiarray as ncm
    g = [(ncm.scalar, "numpy.core.multiarray.scalar"), ncm.scalar, np.dtype]
    g += [type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32")]
    return g


def safe_load(path, mmap=True):
    """torch.load restricted to tensors + numpy scalars (no arbitrary unpickling)."""
    with torch.serialization.safe_globals(_safe_globals()):
        try:
            return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
        except (RuntimeError, ValueError):
            return torch.load(path, map_location="cpu", weights_only=True)


def clean_state(sd):
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    return sd


def load_tile_state(model, path, strict=True, verbose=True):
    ck = safe_load(path)
    sd = ck["tile_model"] if "tile_model" in ck else ck
    sd = clean_state(sd)
    msg = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"[encoder] loaded {path}: missing={list(msg.missing_keys)} unexpected={list(msg.unexpected_keys)}", flush=True)
    if strict and (msg.missing_keys or msg.unexpected_keys):
        raise RuntimeError(f"encoder key mismatch: missing {msg.missing_keys}, unexpected {msg.unexpected_keys}")
    del ck
    return msg


def eagle_lung_path():
    from huggingface_hub import hf_hub_download
    return hf_hub_download("MCCPBR/EAGLE", "tile_model.pth")


def get_encoder(init="gigapath", verbose=True):
    if init == "gigapath":
        try:
            model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        except Exception as e:  # gated repo / no token
            raise SystemExit(
                "\n[encoder] Cannot download prov-gigapath/prov-gigapath (" + type(e).__name__ + ": " + str(e)[:200] + ").\n"
                "This Hugging Face repo is gated. Request access at https://huggingface.co/prov-gigapath/prov-gigapath "
                "with the account whose token is in $HF_HOME, then rerun. "
                "Use --init eagle_lung to start from the public MCCPBR/EAGLE weights instead.\n")
    elif init == "eagle_lung":
        model = build_arch()
        load_tile_state(model, eagle_lung_path(), strict=True, verbose=verbose)
    elif init == "random":
        model = build_arch()
    elif os.path.isfile(init):
        model = build_arch()
        load_tile_state(model, init, strict=True, verbose=verbose)
    else:
        raise ValueError(f"unknown init {init}")
    model.ndim = NDIM
    return model
