import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh


# ============================================================
#  S^3 sampling (Monte-Carlo)
# ============================================================

def sample_on_S3_uniform(n_samples, rng):
    """Uniform samples on S^3 via normalized 4D Gaussians."""
    X = rng.normal(size=(n_samples, 4))
    X /= np.linalg.norm(X, axis=1)[:, None]
    return X


def sample_on_S3_biased(n_samples, alpha, rng, fiber_axis=None):
    """
    Biased sampling on S^3 with density proportional to exp(alpha * cos^2(theta)),
    where cos^2(theta) = (x · fiber_axis)^2.
    Implemented via rejection sampling using an easy envelope.
    """
    if fiber_axis is None:
        fiber_axis = np.array([1.0, 0.0, 0.0, 0.0])

    accepted = []
    w_max = np.exp(alpha)  # since cos^2 in [0, 1]

    while len(accepted) < n_samples:
        v = rng.normal(size=4)
        v /= np.linalg.norm(v)
        cos2 = float(np.dot(v, fiber_axis)) ** 2
        w = np.exp(alpha * cos2)
        if rng.random() < (w / w_max):
            accepted.append(v)

    return np.array(accepted)


def mc_ratio_once(n_samples, alpha, rng, fiber_axis=None):
    """
    Compute the Monte-Carlo ratio:
      R = 8 * <cos^2> / <sin^2>, with sin^2 = 1 - cos^2.
    """
    if fiber_axis is None:
        fiber_axis = np.array([1.0, 0.0, 0.0, 0.0])

    X = sample_on_S3_biased(n_samples, alpha, rng, fiber_axis=fiber_axis)
    cos2 = (X @ fiber_axis) ** 2
    sin2 = 1.0 - cos2
    return float(8.0 * cos2.mean() / sin2.mean())


def mc_curve(alphas, n_samples=100_000, repeats=10, seed=0):
    rng = np.random.default_rng(seed)
    means, cis = [], []

    print("\nMonte-Carlo biased sampling on S^3")
    for a in alphas:
        vals = np.array([mc_ratio_once(n_samples, a, rng) for _ in range(repeats)], dtype=float)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if repeats > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(max(1, repeats))
        means.append(mean)
        cis.append(ci)
        print(f"alpha={a:+.2f} | R_MC={mean:.6f}  95%CI≈±{ci:.6f}")

    return np.array(means), np.array(cis)


# ============================================================
#  Graph + weighted Laplacian (Spectral)
# ============================================================

def knn_edges(X, k=14):
    """Directed kNN edges (i -> j) with squared distances."""
    N = X.shape[0]
    tree = cKDTree(X)
    dists, idxs = tree.query(X, k=k + 1)  # includes self
    rows, cols, dist2 = [], [], []
    for i in range(N):
        for t in range(1, k + 1):
            j = int(idxs[i, t])
            rows.append(i)
            cols.append(j)
            dist2.append(float(dists[i, t] ** 2))
    return np.array(rows), np.array(cols), np.array(dist2)


def make_isotropic_weights(dist2, sigma=None, sigma_scale=1.0):
    eps = 1e-12
    if sigma is None:
        sigma = np.sqrt(np.median(dist2) + eps)
    sig = sigma * sigma_scale
    w = np.exp(-dist2 / (2.0 * sig * sig))
    return w, sigma


def symmetrize_max(A):
    """Make the graph undirected by taking A <- max(A, A^T)."""
    return A.maximum(A.T)


def combinatorial_laplacian(A):
    d = np.array(A.sum(axis=1)).ravel()
    D = sparse.diags(d)
    return (D - A).tocsr()


def fiber_base_energy_ratio(X, A, modes, fiber_axis=None, n_modes_avg=6):
    """
    Spectral analogue:
      R_spec = average_k [ 8 * E_parallel(mode_k) / E_perp(mode_k) ],
    where energies are computed from edge differences, decomposed along fiber axis
    in ambient R^4.
    """
    if fiber_axis is None:
        fiber_axis = np.array([1.0, 0.0, 0.0, 0.0])

    A = A.tocoo()
    i = A.row
    j = A.col
    w = A.data

    # Edge unit directions in ambient R^4
    d = X[j] - X[i]
    dn = np.linalg.norm(d, axis=1)
    dn[dn < 1e-12] = 1.0
    u = d / dn[:, None]

    c = (u @ fiber_axis)
    c2 = c * c
    s2 = 1.0 - c2

    ratios = []
    used = 0
    for m in range(modes.shape[1]):
        f = modes[:, m]
        if np.std(f) < 1e-10:
            continue  # skip constant/near-constant
        df = f[j] - f[i]
        Epar = np.sum(w * c2 * df * df)
        Eper = np.sum(w * s2 * df * df)
        if Eper <= 1e-18:
            continue
        ratios.append(8.0 * Epar / Eper)
        used += 1
        if used >= n_modes_avg:
            break

    return float(np.mean(ratios)) if ratios else np.nan


def spectral_ratio_once(
    N_points, k_nn, alpha, rng,
    gamma=0.5, sigma_scale=1.0,
    n_eigs=18, n_modes_avg=6,
    fiber_axis=None
):
    """
    One spectral estimate at given alpha:
      - sample S^3 points
      - build kNN graph with isotropic Gaussian kernel
      - apply detailed-balance edge reweighting A = S A0 S, S=diag(mu^gamma),
        mu_i = exp(alpha * x0^2)
      - compute lowest eigenmodes of L = D - A
      - return R_spec based on fiber/base edge-energy decomposition
    """
    if fiber_axis is None:
        fiber_axis = np.array([1.0, 0.0, 0.0, 0.0])

    X = sample_on_S3_uniform(N_points, rng)
    rows, cols, dist2 = knn_edges(X, k=k_nn)

    w0, sigma = make_isotropic_weights(dist2, sigma=None, sigma_scale=sigma_scale)
    A0 = sparse.csr_matrix((w0, (rows, cols)), shape=(N_points, N_points))
    A0.sum_duplicates()
    A0.eliminate_zeros()
    A0 = symmetrize_max(A0)

    x2 = X[:, 0] ** 2
    xc2 = 0.25

    if alpha <= 0.0:
      mu = np.ones_like(x2)
    else:
      mu = np.exp(alpha * np.maximum(0.0, x2 - xc2))
    print(f"[debug] alpha={alpha:+.2f} mu: min={mu.min():.4f} max={mu.max():.4f} mean={mu.mean():.4f}")

    s = mu ** gamma
    S = sparse.diags(s)

    # Detailed-balance style reweighting
    A = (S @ A0 @ S).tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()

    L = combinatorial_laplacian(A)

    evals, evecs = eigsh(L, k=n_eigs, which="SM", tol=1e-6)
    idx = np.argsort(np.real(evals))
    evecs = np.real(evecs[:, idx])

    return fiber_base_energy_ratio(X, A, evecs, fiber_axis=fiber_axis, n_modes_avg=n_modes_avg)


def spectral_curve(
    alphas,
    N_points=2500, k_nn=14,
    repeats=6, seed=1,
    gamma=0.5, sigma_scale=1.0,
    n_eigs=18, n_modes_avg=6,
    calibrate_to_8over3=True
):
    rng = np.random.default_rng(seed)
    means, cis = [], []

    print("\nSpectral curve via weighted Laplacian (detailed-balance edges)")
    for a in alphas:
        vals = []
        for _ in range(repeats):
            vals.append(
                spectral_ratio_once(
                    N_points, k_nn, a, rng,
                    gamma=gamma, sigma_scale=sigma_scale,
                    n_eigs=n_eigs, n_modes_avg=n_modes_avg
                )
            )
        vals = np.array(vals, dtype=float)
        mean = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1)) if repeats > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(max(1, repeats))

        means.append(mean)
        cis.append(ci)
        print(f"alpha={a:+.2f} | R_spec={mean:.6f}  95%CI≈±{ci:.6f}")

    means = np.array(means)
    cis = np.array(cis)

    if calibrate_to_8over3:
        # Recalibrate so that R_spec(alpha=0) matches 8/3
        # (use the nearest alpha to 0 in the provided grid)
        idx0 = int(np.argmin(np.abs(alphas)))
        scale = (8.0 / 3.0) / means[idx0]
        means = means * scale
        cis = cis * scale
        print(f"\n[Calibration] Scaled R_spec by factor {scale:.6f} so that R_spec(0)=8/3.")

    return means, cis


# ============================================================
#  Combined plot
# ============================================================

def main():
    alphas = np.linspace(-2.0, 2.0, 17)

    # --- Monte-Carlo settings
    mc_n_samples = 100_000
    mc_repeats = 10

    # --- Spectral settings
    N_points = 2500
    k_nn = 14
    spec_repeats = 6

    # Tunable knobs to better match the MC curve shape
    gamma = 2.4       # try 0.8 or 1.0 for stronger alpha>0 amplification
    sigma_scale = 0.7 # try 0.7 for more locality / stronger density effects

    # Compute curves
    R_mc, CI_mc = mc_curve(alphas, n_samples=mc_n_samples, repeats=mc_repeats, seed=0)
    R_spec, CI_spec = spectral_curve(
        alphas,
        N_points=N_points, k_nn=k_nn,
        repeats=spec_repeats, seed=1,
        gamma=gamma, sigma_scale=sigma_scale,
        n_eigs=18, n_modes_avg=6,
        calibrate_to_8over3=True
    )

    # Plot
    plt.figure(figsize=(10, 6))

    # Reference
    plt.axhline(y=8/3, linestyle="--", label="Theoretical target (8/3)")

    # MC
    plt.errorbar(
        alphas, R_mc, yerr=CI_mc,
        fmt="o-", capsize=4,
        label="Monte-Carlo (biased sampling on S³)"
    )

    # Spectral
    plt.errorbar(
        alphas, R_spec, yerr=CI_spec,
        fmt="o-", capsize=4,
        label=f"Spectral (weighted Laplacian, DB edges)  [gamma={gamma}, sigma_scale={sigma_scale}]"
    )

    plt.xlabel("Relaxation bias α")
    plt.ylabel("Fiber/Base ratio")
    plt.title("Monte-Carlo vs Spectral response on S³ (shared bias parameter α)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
