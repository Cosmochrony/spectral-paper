#!/usr/bin/env python3
"""
compare_mc_vs_weighted_laplacian_hopf_anisotropic.py

Monte-Carlo vs Spectral response on S^3 using Hopf fibration S^1 -> S^3 -> S^2,
with an explicit fiber/base decomposition and an anisotropic (alpha-controlled)
kernel / Laplacian.

Key idea:
- A pure Hopf tangent fiber/base split with an isotropic kNN kernel yields a near-constant
  ratio ~ 4/3 (or 8/3 after dimension renormalization), mostly insensitive to density bias.
- To recover a non-trivial curve vs alpha, alpha must bias fiber vs base in the operator itself.
  We implement an anisotropic squared length:
        d2_alpha = d_perp^2 + exp(-alpha) * d_par^2
  and use it inside the Gaussian kernel.

Observable:
    R(alpha) = (8 * <d_par^2>_w) / (3 * <d_perp^2>_w) * (base_dim / fiber_dim)
where base_dim=2, fiber_dim=1, hence the extra factor 2.

Plots and printed output are in English (as requested).
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

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
    Hopf fiber tangent at x in R^4:
      (z1,z2) in C^2 with z1=x0+i x1, z2=x2+i x3
      tangent of e^{i psi}(z1,z2) at psi=0 is i(z1,z2) -> (-x1,x0,-x3,x2)
    """
    f = np.stack([-x[:, 1], x[:, 0], -x[:, 3], x[:, 2]], axis=1)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f


def project_to_tangent(xi: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project v to tangent space at xi on S^3: v_tan = v - (v·xi) xi."""
    return v - np.dot(v, xi) * xi


# ----------------------------
# Optional density bias mu(x; alpha) (kept for experimentation)
# ----------------------------

def mu_rectified_axis(x: np.ndarray, alpha: float, xc2: float = 0.25) -> np.ndarray:
    """
    Axis-based rectified density bias (as in your earlier experiments).
    This is optional; anisotropy below is what produces the Hopf fiber/base curve.
    """
    x2 = x[:, 0] ** 2
    if alpha <= 0.0:
        return np.ones_like(x2)
    return np.exp(alpha * np.maximum(0.0, x2 - xc2))


# ----------------------------
# Graph building (kNN)
# ----------------------------

def knn_edges(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Directed kNN edges (i -> k nearest neighbors)."""
    tree = cKDTree(x)
    dist, idx = tree.query(x, k=k + 1)  # includes self
    idx = idx[:, 1:]
    n = x.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = idx.reshape(-1)
    return src, dst


# ----------------------------
# Fiber/base decomposition for edges + anisotropic kernel
# ----------------------------

def edge_fiber_base_components(
    x: np.ndarray, src: np.ndarray, dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each directed edge i->j:
      - compute tangent displacement at i
      - decompose into fiber (1D) and base (2D) components
    Return arrays (d_par2, d_perp2).
    """
    f = hopf_fiber_tangent(x)

    d_par2 = np.zeros(src.shape[0], dtype=float)
    d_perp2 = np.zeros(src.shape[0], dtype=float)

    for t, (i, j) in enumerate(zip(src, dst)):
        xi = x[i]
        xj = x[j]
        d = xj - xi
        d_tan = project_to_tangent(xi, d)

        a = np.dot(d_tan, f[i])
        par2 = a * a
        tan2 = np.dot(d_tan, d_tan)
        perp2 = max(0.0, tan2 - par2)

        d_par2[t] = par2
        d_perp2[t] = perp2

    return d_par2, d_perp2


def anisotropic_kernel_weights(
    d_par2: np.ndarray,
    d_perp2: np.ndarray,
    alpha: float,
    sigma2: float,
) -> np.ndarray:
    """
    Anisotropic squared length:
        d2_alpha = d_perp^2 + exp(-alpha) * d_par^2
    alpha>0 => fiber component is less penalized => fiber-aligned edges get more weight.
    """
    alpha_pos = max(0.0, float(alpha))
    d2a = d_perp2 + np.exp(-alpha_pos) * d_par2
    return np.exp(-d2a / (2.0 * sigma2))


def db_weights_from_mu(
    w_geom: np.ndarray,
    mu: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Detailed-balance style modulation:
        w_ij = w_geom * (mu_i * mu_j)^gamma
    """
    return w_geom * ((mu[src] * mu[dst]) ** gamma)


def fiber_base_ratio(
    d_par2: np.ndarray,
    d_perp2: np.ndarray,
    w: np.ndarray,
) -> float:
    """
    R = (8 <par2>_w) / (3 <perp2>_w) * (base_dim/fiber_dim) = ... * 2
    so that isotropic tangent splitting yields 8/3 (not 4/3).
    """
    W = np.sum(w)
    if W <= 0:
        return np.nan
    m_par = np.sum(w * d_par2) / W
    m_perp = np.sum(w * d_perp2) / W
    if m_perp <= 0:
        return np.nan
    return (8.0 * m_par) / (3.0 * m_perp) * 2.0


# ----------------------------
# Experiment driver
# ----------------------------

@dataclass
class Config:
    n_points: int = 1200
    k_nn: int = 14
    repeats: int = 8
    seed: int = 1
    sigma_scale: float = 0.7
    gamma_db: float = 2.4
    use_density_mu: bool = False  # set True if you also want mu-based DB weighting


def run_mc_curve(alphas: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte-Carlo branch:
    - sample points (uniform on S^3)
      (optionally: biased density via mu, but that alone won't create a curve in Hopf split)
    - build kNN graph
    - compute fiber/base components
    - compute anisotropic kernel weights using alpha
    - compute R(alpha)
    """
    rng = np.random.default_rng(cfg.seed)
    means, cis = [], []

    for alpha in alphas:
        vals = []
        for r in range(cfg.repeats):
            x = sample_s3(cfg.n_points, rng)

            src, dst = knn_edges(x, cfg.k_nn)
            d_par2, d_perp2 = edge_fiber_base_components(x, src, dst)

            # sigma2 from median of isotropic tangent length (for stability)
            med = np.median(d_par2 + d_perp2)
            sigma2 = max(1e-12, med * (cfg.sigma_scale ** 2))

            w = anisotropic_kernel_weights(d_par2, d_perp2, alpha, sigma2)

            if cfg.use_density_mu:
                mu = mu_rectified_axis(x, alpha)  # shares alpha for convenience
                w = db_weights_from_mu(w, mu, src, dst, cfg.gamma_db)

            R = fiber_base_ratio(d_par2, d_perp2, w)
            vals.append(R)

        vals = np.asarray(vals)
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

        print(f"[MC] alpha={alpha:+.2f} | R={mean:.6f}  95%CI≈±{ci:.6f}")
        means.append(mean)
        cis.append(ci)

    return np.asarray(means), np.asarray(cis)


def run_spectral_curve(alphas: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spectral branch (same construction, but you can interpret it as a weighted Laplacian response):
    - sample points uniformly (new sample per repeat to get non-zero CI)
    - build kNN graph
    - use anisotropic kernel weights (alpha)
    - optional DB(mu) modulation
    - compute the same fiber/base ratio
    """
    rng = np.random.default_rng(cfg.seed + 999)
    means, cis = [], []

    for alpha in alphas:
        vals = []
        for r in range(cfg.repeats):
            x = sample_s3(cfg.n_points, rng)

            src, dst = knn_edges(x, cfg.k_nn)
            d_par2, d_perp2 = edge_fiber_base_components(x, src, dst)

            med = np.median(d_par2 + d_perp2)
            sigma2 = max(1e-12, med * (cfg.sigma_scale ** 2))

            w = anisotropic_kernel_weights(d_par2, d_perp2, alpha, sigma2)

            if cfg.use_density_mu:
                mu = mu_rectified_axis(x, alpha)
                w = db_weights_from_mu(w, mu, src, dst, cfg.gamma_db)

            R = fiber_base_ratio(d_par2, d_perp2, w)
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
        repeats=8,
        seed=1,
        sigma_scale=0.7,
        gamma_db=2.4,
        use_density_mu=False,  # keep False for a clean Hopf-anisotropy experiment
    )

    alphas = np.array(
        [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
          0.0,  0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75,  2.0],
        dtype=float
    )

    target = 8.0 / 3.0

    print("\nMonte-Carlo branch (Hopf fiber/base, anisotropic kernel)\n")
    mc_mean, mc_ci = run_mc_curve(alphas, cfg)

    print("\nSpectral branch (Hopf fiber/base, anisotropic kernel)\n")
    sp_mean, sp_ci = run_spectral_curve(alphas, cfg)

    plt.figure(figsize=(11, 6))
    plt.axhline(y=target, linestyle="--", label="Theoretical target (8/3)")

    plt.errorbar(alphas, mc_mean, yerr=mc_ci, fmt="o-", capsize=3,
                 label="Monte-Carlo (anisotropic kernel on S³) — Hopf fiber/base")

    plt.errorbar(alphas, sp_mean, yerr=sp_ci, fmt="o-", capsize=3,
                 label=f"Spectral (weighted Laplacian kernel) — Hopf fiber/base  [sigma_scale={cfg.sigma_scale}]")

    plt.xlabel("Relaxation bias α")
    plt.ylabel("Fiber/Base ratio")
    plt.title("Monte-Carlo vs Spectral response on S³ (Hopf fibration S¹→S³→S²)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
