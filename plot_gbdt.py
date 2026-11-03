import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = pd.read_csv("results_gbdt.csv")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for m, c in [("SHAP", "#c0392b"), ("Permutation", "#2471a3")]:
    g = d[d.method == m].groupby("rho")
    for a, col, title in [(ax[0], "proxy_share", "Attribution leaked to non-causal proxy"),
                          (ax[1], "true_share", "Attribution on planted true cells")]:
        mu, sd = g[col].mean(), g[col].std()
        a.errorbar(mu.index.astype(str), mu.values, yerr=sd.values, marker="o", capsize=3, label=m, color=c)
        a.set_title(title, fontsize=10); a.set_xlabel("correlation of proxy with true driver (rho)")
        a.set_ylabel("share of total attribution")
ax[0].legend(); plt.tight_layout(); plt.savefig("fig_gbdt_attribution.png", dpi=160)
print(d.groupby("rho").r2.mean().round(3).to_dict())
