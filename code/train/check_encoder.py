"""CPU check: explicit ViT-g config == MCCPBR/EAGLE tile_model.pth keys/shapes (strict), plus a forward pass through
the released EAGLE (lung) encoder + aggregator."""
import sys, torch
sys.path.insert(0, ".")
import encoder as E, modules
from huggingface_hub import hf_hub_download
m = E.build_arch()
print("arch params", sum(p.numel() for p in m.parameters()))
ck = E.safe_load(E.eagle_lung_path()); sd = E.clean_state(ck["tile_model"])
ms = m.state_dict()
print("ckpt keys", len(sd), "model keys", len(ms), "missing", sorted(set(ms) - set(sd)), "unexpected", sorted(set(sd) - set(ms)))
print("shape mismatches", [(k, tuple(sd[k].shape), tuple(ms[k].shape)) for k in ms if k in sd and sd[k].shape != ms[k].shape])
E.load_tile_state(m, E.eagle_lung_path(), strict=True)
g = modules.GMA(ndim=1536, dropout=True); c2 = E.safe_load(hf_hub_download("MCCPBR/EAGLE", "slide_model.pth"), mmap=False)
print("gma load:", g.load_state_dict(c2["slide_model"]))
m.eval(); g.eval(); torch.manual_seed(0)
with torch.no_grad():
    h = m(torch.randn(4, 3, 224, 224)); _, _, o = g(h)
print("features", tuple(h.shape), "mean", float(h.mean()), "std", float(h.std()), "EAGLE-lung P(EGFR) on noise", float(torch.softmax(o, 1)[0, 1]))
print("ENCODER_CHECK_OK")
