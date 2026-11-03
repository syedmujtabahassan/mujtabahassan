import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = pd.read_csv("results_gbdt_noisy.csv")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
cols = [("r2", "Model accuracy (R$^2$)"), ("proxy_share", "Attribution on non-causal proxy"),
        ("true_share", "Attribution on planted true cells")]
for m, c in [("SHAP", "#c0392b"), ("Permutation", "#2471a3")]:
    g = d[d.method == m].groupby("rho")
    for a, (col, title) in zip(ax, cols):
        mu, sd = g[col].mean(), g[col].std()
        a.errorbar(mu.index.astype(str), mu.values, yerr=sd.values, marker="o", capsize=3, label=m, color=c)
        a.set_title(title, fontsize=10); a.set_xlabel("proxy correlation rho (traffic sensor noise = 1.0)")
ax[0].legend(); plt.tight_layout(); plt.savefig("fig_gbdt_noisy_sensor.png", dpi=160)
