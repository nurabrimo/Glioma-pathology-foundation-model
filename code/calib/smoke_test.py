"""Smoke test on the authors' example_data.csv: (1) ported functions reproduce the original utils (run when seaborn is
available), (2) simulation_fast == simulation, (3) evaluate.py end to end on a synthetic pretrial/silent split."""
import os, subprocess, sys, tempfile
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import utils as U
df = pd.read_csv(os.path.join(here, "example_data.csv")).dropna().reset_index(drop=True)
df["target"] = df.target.astype(int); df["rapid"] = df.rapid.astype(int)
print("example rows", len(df), "pos", int(df.target.sum()), "rapid NPV/PPV", round(U.npv_score(df.target, df.rapid), 4), round(U.ppv_score(df.target, df.rapid), 4))
X, Y, npv, ppv, sen, spe, bac, red = U.simulation(df, N=21)
X2, Y2, npv2, ppv2, sen2, spe2, bac2, red2 = U.simulation_fast(df, np.linspace(0, .5, 21), np.linspace(.5, 1, 21))
d = max(np.nanmax(np.abs(a - b)) for a, b in [(npv, npv2), (ppv, ppv2), (sen, sen2), (spe, spe2), (red, red2), (X, X2), (Y, Y2)])
assert all((np.isnan(a) == np.isnan(b)).all() for a, b in [(npv, npv2), (ppv, ppv2)])
print("simulation_fast vs simulation max abs diff:", d); assert d < 1e-12
try:
    import importlib.util, matplotlib; matplotlib.use("Agg")
    sp = importlib.util.spec_from_file_location("orig", os.path.join(here, "eagle_calib_orig.py")); O = importlib.util.module_from_spec(sp); sp.loader.exec_module(O)
    a = O.get_performance_assisted(df, th0=[0.023], th1=[0.997]); b = U.get_performance_assisted(df, th0=[0.023], th1=[0.997])
    print("original:", a.round(4).to_dict("records")[0]); print("port    :", b.round(4).to_dict("records")[0])
    assert np.allclose(a.values.astype(float), b.values.astype(float))
    Xo = O.simulation(df, N=11); print("original simulation equal:", all(np.allclose(p, q, equal_nan=True) for p, q in zip(Xo[:6], U.simulation(df, N=11)[:6])))
except ImportError as e:
    print("original utils not importable here (", e, "): comparison against the original skipped")
print(U.get_performance_assisted_bootstrapped(df, th0=[0.023], th1=[0.997], n=200, seed=0).round(4).to_dict("records")[0])
tmp = tempfile.mkdtemp(); rs = np.random.RandomState(0)
sd = pd.DataFrame(dict(slide=[f"s{i}" for i in range(len(df))], IDH=df.target, rapid=df.rapid, split=np.where(rs.rand(len(df)) < .5, "pretrial", "silent"),
                       cohort="tcga", institution=rs.choice(list("ABC"), len(df)), idh_variant=np.where(df.target == 1, np.where(df.rapid == 1, "R132H", "other"), "")))
sd.to_csv(tmp + "/slide_data.csv", index=False)
pd.DataFrame(dict(slide=sd.slide, score=df.score, n_tiles=rs.randint(500, 50000, len(df)))).to_csv(tmp + "/scores.csv", index=False)
subprocess.check_call([sys.executable, os.path.join(here, "evaluate.py"), "--scores", tmp + "/scores.csv", "--slide_data", tmp + "/slide_data.csv", "--out", tmp + "/eval", "--nboot", "200", "--min_n", "10"])
print(sorted(os.listdir(tmp + "/eval"))); print("CALIB_SMOKE_OK")
