"""Experiment 1 (no torch needed): do SHAP / permutation importance recover planted drivers
for a GBDT as the proxy-correlation rho grows?  Writes results_gbdt.csv"""
import numpy as np, pandas as pd, shap, lightgbm as lgb
from sklearn.metrics import average_precision_score, r2_score
from datagen import make_series, make_windows, truth_mask, F

L = 8
MASK = truth_mask(L)
rows = []
for rho in [0.0, 0.5, 0.9, 0.99]:
    for seed in range(5):
        X, y = make_series(rho=rho, seed=seed)
        W, Y = make_windows(X, y, L)
        n = int(0.7 * len(Y))
        Xtr, Ytr = W[:n].reshape(n, -1), Y[:n]
        Xte, Yte = W[n:].reshape(len(Y) - n, -1), Y[n:]
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                              subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=seed)
        m.fit(Xtr, Ytr)
        r2 = r2_score(Yte, m.predict(Xte))

        # --- SHAP (TreeExplainer) on a test subsample
        sub = Xte[:400]
        sv = shap.TreeExplainer(m).shap_values(sub)
        shap_attr = np.abs(sv).mean(0).reshape(L, F)

        # --- permutation importance (model-agnostic), per cell
        base = np.mean((Yte - m.predict(Xte)) ** 2)
        rng = np.random.default_rng(seed)
        perm = np.zeros(L * F)
        for c in range(L * F):
            Xp = Xte.copy()
            Xp[:, c] = rng.permutation(Xp[:, c])
            perm[c] = np.mean((Yte - m.predict(Xp)) ** 2) - base
        perm_attr = np.clip(perm, 0, None).reshape(L, F)

        for name, A in [("SHAP", shap_attr), ("Permutation", perm_attr)]:
            tot = A.sum() + 1e-12
            rows.append(dict(rho=rho, seed=seed, method=name, model="GBDT", r2=r2,
                             auprc=average_precision_score(MASK.ravel(), A.ravel()),
                             proxy_share=A[:, 5].sum() / tot,
                             traffic_share=A[:, 1].sum() / tot,
                             true_share=A[MASK == 1].sum() / tot))
    print("done rho", rho, flush=True)
pd.DataFrame(rows).to_csv("results_gbdt.csv", index=False)
print(pd.DataFrame(rows).groupby(["method", "rho"])[["r2", "auprc", "proxy_share", "traffic_share", "true_share"]].mean().round(3))
