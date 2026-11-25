"""Planning power for the paired Hard-vs-teacher comparison. Outcomes: per opening, each version plays the teacher
as Red and as Black. Old Hard's per-colour W/D rates are entry 65 arm P's (Red 18/6/86, Black 5/0/105). The new
version's win probabilities are scaled by k so the mean score rises by delta. A Gaussian copula with a shared
(opening, colour) latent gives correlation rho between the two versions' games at the same opening and colour."""
import math, sys
import numpy as np
from statistics import NormalDist
IQ = NormalDist().inv_cdf

RED_W, RED_D, BLK_W, BLK_D = 18/110, 6/110, 5/110, 0.0
BASE = (RED_W + RED_D/2 + BLK_W + BLK_D/2) / 2

def outcome(x, pw, pd):
    t1 = IQ(min(max(pw, 1e-12), 1 - 1e-12)); t2 = IQ(min(max(pw + pd, 1e-12), 1 - 1e-12))
    return np.where(x < t1, 1.0, np.where(x < t2, 0.5, 0.0))

def sim(n, delta, rho, rng):
    k = 1 + delta / ((RED_W + BLK_W) / 2)
    z = rng.standard_normal((n, 2))
    e = rng.standard_normal((n, 2, 2))
    u = math.sqrt(rho) * z[:, :, None] + math.sqrt(1 - rho) * e   # [opening, colour, version]
    s = np.empty((n, 2, 2))
    for v, kk in ((0, 1.0), (1, k)):
        s[:, 0, v] = outcome(u[:, 0, v], RED_W * kk, RED_D)
        s[:, 1, v] = outcome(u[:, 1, v], BLK_W * kk, BLK_D)
    return s.mean(axis=1)          # [opening, version] opening score

def sign_p(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    m = pos + neg
    return 1.0 if m == 0 else sum(math.comb(m, j) for j in range(pos, m + 1)) / 2 ** m

def signflip_p(d):
    a = np.rint(np.abs(d) * 4).astype(int); a = a[a > 0]
    obs = int(np.rint(d.sum() * 4))
    tot = int(a.sum()); dist = np.zeros(2 * tot + 1); dist[tot] = 1.0
    for x in a:
        nd = np.zeros_like(dist); nd[x:] += dist[:len(dist) - x] / 2; nd[:len(dist) - x] += dist[x:] / 2; dist = nd
    return float(dist[obs + tot:].sum()) if tot else 1.0

def power(n, delta, rho, reps, seed):
    rng = np.random.default_rng(seed); ps = pf = 0
    for _ in range(reps):
        s = sim(n, delta, rho, rng); d = s[:, 1] - s[:, 0]
        ok_mean = d.mean() > 0
        ps += (sign_p(d) <= 0.05 and ok_mean); pf += (signflip_p(d) <= 0.05 and ok_mean)
    return ps / reps, pf / reps

if __name__ == "__main__":
    print("base score", round(BASE, 4))
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    for rho in (0.0, 0.36, 0.6):
        for delta in (0.0, 0.04, 0.05, 0.06, 0.08):
            row = []
            for n in (150, 200, 250, 300, 400):
                a, b = power(n, delta, rho, reps, 20261124 + n)
                row.append(f"n{n}: sign {a:.2f} flip {b:.2f}")
            print(f"rho {rho} delta {delta}: " + " | ".join(row), flush=True)
