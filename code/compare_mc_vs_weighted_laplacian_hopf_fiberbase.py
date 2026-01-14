#!/usr/bin/env python3
"""
compare_mc_vs_weighted_laplacian_hopf_fiberbase.py

Monte-Carlo vs Spectral response on S^3 (Hopf fiber/base decomposition)

We compare two ways of introducing the same bias parameter alpha:

1) Monte-Carlo (biased sampling on S^3):
   - Draw points on S^3 with acceptance weight mu(x; alpha)
   - Build an unweighted kNN graph (geometric weights only)
   - Measure fiber/base edge energy using Hopf fiber tangent f(x)

2) Spectral (weighted Laplacian, detailed-balance edges):
   - Draw points uniformly on S^3
   - Build a kNN graph
   - Apply DB weights using mu(x; alpha) and a Gaussian kernel
   - Measure the same fiber/base edge energy on that weighted graph

Observable:
  R(alpha) = (8 * <||d_parallel||^2>_w) / (3 * <||d_perp||^2>_w)

Where d is the edge displacement projected to the tangent space of S^3 at the source node,
d_parallel is the component along the Hopf fiber tangent f(x),
and d_perp is the remaining tangent component (base directions).

Notes:
- Code and plots are in English (as requested).
- You can tune gamma and sigma_scale for the spectral (DB) weighting.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List

from scipy.spatial import cKDTree


# ----------------------------
# Geometry: S^3 + Hopf fiber
# ----------------------------

def sample_s3(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample on S^3 via normalized Gaussian in R^4."""
    x = rng.normal(size=(n, 4))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def hopf_fiber_tangent(x: np.ndarray) -> np.ndarray:
    """
    Hopf fiber tangent at x in R^4 representation:
    Let (z1, z2) in C^2 with z1 = x0 + i x1, z2 = x2 + i x3.
    Fiber action: e^{i psi} (z1, z2).
    Tangent: d/dpsi at psi=0 gives i(z1, z2) => (-x1, x0, -x3, x2) in R^4.
    This vector is tangent to S^3 and has unit norm if x is on S^3.
    """
    f = np.stack([-x[:, 1], x[:, 0], -x[:, 3], x[:, 2]], axis=1)
    # numerical normalize (should already be norm 1)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f


def project_to_tangent(xi: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Project v to the tangent space of S^3 at xi: v_tan = v - (v·xi) xi
    """
    return v - np.dot(v, xi) * xi


# ----------------------------
# Bias / measure mu(x; alpha)
# ----------------------------

def mu_rectified(x: np.ndarray, alpha: float, xc2: float = 0.25) -> np.ndarray:
    """
    Rectified bias used previously:
      if alpha <= 0: mu = 1  (no active repulsion; keeps negative branch flat)
      else: mu = exp(alpha * max(0, x0^2 - xc2))
    """
    x2 = x[:, 0] ** 2
    if alpha <= 0.0:
        return np.ones_like(x2)
    return np.exp(alpha * np.maximum(0.0, x2 - xc2))


# ----------------------------
# Graph building (kNN)
# ----------------------------

def knn_edges(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build directed kNN edges: for each i, connect to k nearest neighbors (excluding itself).
    Returns (src, dst, dist2) arrays of length n*k.
    """
    tree = cKDTree(x)
    dist, idx = tree.query(x, k=k + 1)  # includes self at position 0
    idx = idx[:, 1:]
    dist = dist[:, 1:]
    n = x.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = idx.reshape(-1)
    dist2 = (dist.reshape(-1) ** 2)
    return src, dst, dist2


def gaussian_kernel(dist2: np.ndarray, sigma2: float) -> np.ndarray:
    return np.exp(-dist2 / (2.0 * sigma2))


# ----------------------------
# Detailed-balance edge weights
# ----------------------------

def db_edge_weights(
    src: np.ndarray,
    dst: np.ndarray,
    dist2: np.ndarray,
    mu: np.ndarray,
    sigma2: float,
    gamma: float,
) -> np.ndarray:
    """
    Detailed-balance symmetric edge weights (for an undirected effective operator)
    but implemented on directed edge lists by using symmetric form:
      w_ij = K_ij * (mu_i * mu_j)^gamma
    with K_ij = exp(-||xi-xj||^2 / (2 sigma^2)).
    """
    K = gaussian_kernel(dist2, sigma2)
    w = K * ((mu[src] * mu[dst]) ** gamma)
    return w


# ----------------------------
# Hopf fiber/base edge energy
# ----------------------------

def fiber_base_energy_ratio(
    x: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
) -> float:
    """
    Compute R = (8 * <||d_parallel||^2>_w) / (3 * <||d_perp||^2>_w)
    where d is the tangent-projected displacement at the source node,
    decomposed along the Hopf fiber tangent f(x_i).
    """
    f = hopf_fiber_tangent(x)  # (n,4)

    E_par = 0.0
    E_perp = 0.0
    Wsum = 0.0

    for i, j, wij in zip(src, dst, w):
        xi = x[i]
        xj = x[j]
        d = xj - xi
        d_tan = project_to_tangent(xi, d)

        # fiber component
        a = np.dot(d_tan, f[i])
        d_par2 = a * a

        d_tan2 = np.dot(d_tan, d_tan)
        d_perp2 = max(0.0, d_tan2 - d_par2)

        E_par += wij * d_par2
        E_perp += wij * d_perp2
        Wsum += wij

    if Wsum <= 0:
        return np.nan

    # weighted means
    m_par = E_par / Wsum
    m_perp = E_perp / Wsum
    if m_perp <= 0:
        return np.nan

    return (8.0 * m_par) / (3.0 * m_perp)


# ----------------------------
# Experiment driver
# ----------------------------

@dataclass
class Config:
    n_points: int = 1200
    k_nn: int = 14
    sigma_scale: float = 0.7
    gamma: float = 2.4
    repeats: int = 6
    seed: int = 1


def run_curve_mc_biased_sampling(alphas: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte-Carlo branch:
    - sample points with acceptance probability proportional to mu (biased sampling)
    - build unweighted kNN graph with geometric Gaussian weights only
    - compute Hopf fiber/base ratio
    """
    rng = np.random.default_rng(cfg.seed)
    means = []
    cis = []

    for alpha in alphas:
        vals = []
        for r in range(cfg.repeats):
            # rejection sampling: oversample then accept by mu/max(mu)
            overs = int(cfg.n_points * 3.0)
            x0 = sample_s3(overs, rng)
            mu0 = mu_rectified(x0, alpha)
            mmax = float(mu0.max())
            accept = rng.random(size=overs) < (mu0 / mmax)
            x = x0[accept]
            if x.shape[0] < cfg.n_points:
                # fallback: top up if acceptance too low
                needed = cfg.n_points - x.shape[0]
                x_add = sample_s3(needed, rng)
                x = np.vstack([x, x_add])
            else:
                x = x[: cfg.n_points]

            src, dst, dist2 = knn_edges(x, cfg.k_nn)

            # geometric weights only (no mu here, because bias is in sampling)
            # sigma2 is set from median dist2 * sigma_scale^2 for stability
            med = np.median(dist2)
            sigma2 = max(1e-12, med * (cfg.sigma_scale ** 2))
            w = gaussian_kernel(dist2, sigma2)

            R = fiber_base_energy_ratio(x, src, dst, w)
            vals.append(R)

        vals = np.asarray(vals)
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

        print(f"[MC] alpha={alpha:+.2f} | R={mean:.6f}  95%CI≈±{ci:.6f}")
        means.append(mean)
        cis.append(ci)

    return np.asarray(means), np.asarray(cis)


def run_curve_spectral_db_weights(alphas: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spectral branch:
    - sample points uniformly
    - build kNN graph
    - apply detailed-balance weights using mu(x; alpha)
    - compute Hopf fiber/base ratio on weighted graph
    """
    rng = np.random.default_rng(cfg.seed + 12345)
    means = []
    cis = []

    # fixed geometry (optional): keep same point cloud across alphas for stability
    base_x = sample_s3(cfg.n_points, rng)
    base_src, base_dst, base_dist2 = knn_edges(base_x, cfg.k_nn)
    base_med = np.median(base_dist2)
    base_sigma2 = max(1e-12, base_med * (cfg.sigma_scale ** 2))

    for alpha in alphas:
        vals = []
        for r in range(cfg.repeats):
            # for repeats, we can resample only if desired; keep fixed to reduce noise
            x = base_x
            src, dst, dist2 = base_src, base_dst, base_dist2

            mu = mu_rectified(x, alpha)
            w = db_edge_weights(src, dst, dist2, mu, base_sigma2, cfg.gamma)

            R = fiber_base_energy_ratio(x, src, dst, w)
            vals.append(R)

        vals = np.asarray(vals)
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

        print(f"[Spec] alpha={alpha:+.2f} | R={mean:.6f}  95%CI≈±{ci:.6f}")
        means.append(mean)
        cis.append(ci)

    return np.asarray(means), np.asarray(cis)


def main():
    cfg = Config(
        n_points=1200,
        k_nn=14,
        sigma_scale=0.7,
        gamma=2.4,
        repeats=6,
        seed=1,
    )

    alphas = np.array(
        [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
          0.0,  0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75,  2.0],
        dtype=float
    )

    print("\nMonte-Carlo biased sampling on S^3 (Hopf fiber/base)\n")
    mc_mean, mc_ci = run_curve_mc_biased_sampling(alphas, cfg)

    print("\nSpectral response via weighted Laplacian (DB edges) (Hopf fiber/base)\n")
    sp_mean, sp_ci = run_curve_spectral_db_weights(alphas, cfg)

    # Optional: calibrate spectral so that alpha=0 matches 8/3 (same as your previous workflow)
    target = 8.0 / 3.0
    idx0 = int(np.where(alphas == 0.0)[0][0])
    scale = target / sp_mean[idx0]
    sp_mean_cal = sp_mean * scale
    sp_ci_cal = sp_ci * scale
    print(f"\n[Calibration] Scaled spectral curve by factor {scale:.6f} so that R_spec(0)=8/3.\n")

    # Plot (English)
    plt.figure(figsize=(11, 6))
    plt.axhline(y=target, linestyle="--", label="Theoretical target (8/3)")

    plt.errorbar(alphas, mc_mean, yerr=mc_ci, fmt="o-", capsize=3,
                 label="Monte-Carlo (biased sampling on S³) — Hopf fiber/base")

    plt.errorbar(alphas, sp_mean_cal, yerr=sp_ci_cal, fmt="o-", capsize=3,
                 label=f"Spectral (weighted Laplacian, DB edges) — Hopf fiber/base  [gamma={cfg.gamma}, sigma_scale={cfg.sigma_scale}]")

    plt.xlabel("Relaxation bias α")
    plt.ylabel("Fiber/Base ratio")
    plt.title("Monte-Carlo vs Spectral response on S³ (Hopf fibration S¹→S³→S²)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
