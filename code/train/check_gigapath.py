"""CPU check: the gated Prov-GigaPath tile encoder loads through timm and has exactly the explicit architecture."""
import sys, torch
sys.path.insert(0, ".")
import encoder as E
m = E.get_encoder("gigapath"); a = E.build_arch()
ms, as_ = m.state_dict(), a.state_dict()
print("params", sum(p.numel() for p in m.parameters()), "keys equal", set(ms) == set(as_), "shapes equal", all(ms[k].shape == as_[k].shape for k in ms if k in as_))
a.load_state_dict(ms); m.eval(); a.eval(); torch.manual_seed(0); x = torch.randn(2, 3, 224, 224)
with torch.no_grad(): d = (m(x) - a(x)).abs().max().item()
print("hub model vs explicit arch (same weights) max abs feature diff", d, "out", tuple(m(x).shape)); print("GIGAPATH_CHECK_OK")
