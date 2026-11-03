"""
Synthetic environmental-style time series with PLANTED ground-truth drivers.

Features (F=6): wind, traffic, temperature, humidity, pressure, proxy
  - wind, traffic, temperature  -> TRUE drivers of the target (at known lags)
  - humidity, pressure          -> pure distractors (independent noise-like)
  - proxy                       -> NON-causal distractor, correlated (rho) with traffic

Target (pollutant level):
  y_t = -0.8*wind[t-3] + 0.9*traffic[t-1] + 0.5*tanh(temp[t-2])
        + 0.3*wind[t-3]*traffic[t-1] + noise

The model sees a window of the previous L steps and predicts y_t.
Ground-truth mask M[j, f] = 1 for the (window position, feature) cells used by y_t.
"""
import numpy as np

FEATURES = ["wind", "traffic", "temp", "humidity", "pressure", "proxy"]
F = len(FEATURES)
TRUE_CELLS_LAG = {"wind": 3, "traffic": 1, "temp": 2}  # feature -> lag (steps before t)


def _ar1(n, phi, rng):
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi**2) * e[t]
    return x


def make_series(n=6000, rho=0.9, noise=0.3, seed=0, obs_noise=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    wind = _ar1(n, 0.8, rng)
    traffic = 0.7 * np.sin(2 * np.pi * t / 24) + 0.7 * _ar1(n, 0.6, rng)
    temp = 0.8 * np.sin(2 * np.pi * t / 240) + 0.6 * _ar1(n, 0.9, rng)
    humidity = _ar1(n, 0.85, rng)
    pressure = _ar1(n, 0.95, rng)

    def z(a):
        return (a - a.mean()) / a.std()

    wind, traffic, temp, humidity, pressure = map(z, (wind, traffic, temp, humidity, pressure))
    # proxy: correlated with traffic at the SAME time step, corr ~= rho, otherwise independent
    indep = z(_ar1(n, 0.6, rng))
    proxy = rho * traffic + np.sqrt(1 - rho**2) * indep

    X = np.stack([wind, traffic, temp, humidity, pressure, proxy], axis=1)  # (n, F)
    y = np.zeros(n)
    for i in range(3, n):
        y[i] = (-0.8 * wind[i - 3] + 0.9 * traffic[i - 1] + 0.5 * np.tanh(temp[i - 2])
                + 0.3 * wind[i - 3] * traffic[i - 1])
    y = y + noise * rng.standard_normal(n)
    if obs_noise > 0:
        # sensor noise on the OBSERVED traffic feature only; the target (and the proxy)
        # were generated from the clean traffic, so the proxy can carry information the
        # noisy sensor lost. Drawn last so default (obs_noise=0) streams are unchanged.
        X[:, 1] = X[:, 1] + obs_noise * rng.standard_normal(n)
    return X.astype(np.float32), y.astype(np.float32)


def make_windows(X, y, L=8):
    """Window j (0..L-1) of sample t holds X[t-L+j]; lag k = L - j."""
    xs, ys = [], []
    for t in range(L, len(y)):
        xs.append(X[t - L:t])
        ys.append(y[t])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def truth_mask(L=8):
    M = np.zeros((L, F), dtype=np.float32)
    for f, lag in TRUE_CELLS_LAG.items():
        M[L - lag, FEATURES.index(f)] = 1.0
    return M


if __name__ == "__main__":
    X, y = make_series(rho=0.9)
    print("corr(traffic, proxy) =", np.corrcoef(X[:, 1], X[:, 5])[0, 1].round(3))
    W, Y = make_windows(X, y)
    print("windows", W.shape, "targets", Y.shape, "target std", Y.std().round(3))
    print(truth_mask())
    # sanity: a linear model on the flattened window should already do well,
    # and its coefficients should concentrate on the true cells when rho is low
    from sklearn.linear_model import Ridge
    for rho in [0.0, 0.9, 0.99]:
        X, y = make_series(rho=rho)
        W, Y = make_windows(X, y)
        n = int(0.8 * len(Y))
        m = Ridge(alpha=1.0).fit(W[:n].reshape(n, -1), Y[:n])
        r2 = m.score(W[n:].reshape(len(Y) - n, -1), Y[n:])
        coef = np.abs(m.coef_.reshape(8, F))
        mask = truth_mask()
        print(f"rho={rho}: ridge R2={r2:.3f}, coef mass on true cells={coef[mask==1].sum()/coef.sum():.2f}, "
              f"on proxy={coef[:, 5].sum()/coef.sum():.2f}")
