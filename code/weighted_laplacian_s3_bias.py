import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


# -----------------------------
# Geometry / sampling
# -----------------------------

def sample_S3(N, rng):
    """Uniform samples on S^3 via normalized 4D Gaussians."""
    X = rng.normal(size=(N, 4))
    X /= np.linalg.norm(X, axis=1)[:, None]
    return X

def chord_dist2(a, b):
    """Squared chord distance on S^3 (in R^4)."""
    d = a - b
    return np.dot(d, d)

def unit_dir(a, b, eps=1e-12):
    """Unit direction from a to b in ambient R^4."""
    d = b - a
    n = np.linalg.norm(d)
    if n < eps:
        return np.zeros_like(d)
    return d / n


# -----------------------------
# Graph construction
# -----------------------------

def knn_graph(X, k=12):
    """
    Build an undirected k-NN graph (symmetric) from points X in R^4.
    Returns neighbor lists (i, j, dist2_ij).
    """
    N = X.shape[0]
    tree = cKDTree(X)
    dists, idxs = tree.query(X, k=k+1)  # includes self at position 0
    # Flatten edges (i -> neighbors)
    rows = []
    cols = []
    dist2 = []
    for i in range(N):
        for t in range(1, k+1):  # skip self
            j = int(idxs[i, t])
            rows.append(i)
            cols.append(j)
            dist2.append(float(dists[i, t] ** 2))
    return np.array(rows), np.array(cols), np.array(dist2)


def make_weighted_adjacency(X, rows, cols, dist2, alpha,
                            fiber_axis=np.array([1.0, 0.0, 0.0, 0.0]),
                            sigma=None,
                            base_kernel="gaussian",
                            anisotropy="directional",
                            symmetrize=True):
    """
    Build a sparse weighted adjacency A_alpha.

    base_kernel:
      - "gaussian": w0 = exp(-dist2/(2*sigma^2))
      - "inverse":  w0 = 1/(dist2 + eps)

    anisotropy:
      - "directional": multiply by exp(alpha * (u·fiber)^2)
         where u is the unit direction along the edge in ambient R^4.
         This is the closest analogue to your exp(alpha cos^2 θ).

    If symmetrize=True, the graph is made undirected by A <- max(A, A^T).
    """
    N = X.shape[0]
    eps = 1e-12

    if sigma is None:
        # robust scale from median neighbor distance
        sigma = np.sqrt(np.median(dist2) + eps)

    # Base isotropic kernel
    if base_kernel == "gaussian":
        w0 = np.exp(-dist2 / (2.0 * sigma * sigma))
    elif base_kernel == "inverse":
        w0 = 1.0 / (dist2 + eps)
    else:
        raise ValueError("Unknown base_kernel")

    # Anisotropy factor
    if anisotropy == "directional":
        # compute (u·fiber)^2 for each directed edge
        dotsq = np.empty_like(w0)
        for t in range(len(rows)):
            i = rows[t]
            j = cols[t]
            u = unit_dir(X[i], X[j])
            c = float(np.dot(u, fiber_axis))
            dotsq[t] = c * c
        w = w0 * np.exp(alpha * dotsq)
    else:
        raise ValueError("Unknown anisotropy")

    A = sparse.csr_matrix((w, (rows, cols)), shape=(N, N))
    A.sum_duplicates()
    A.eliminate_zeros()

    if symmetrize:
        # make undirected: keep the stronger connection each way
        A = A.maximum(A.T)

    return A, sigma


# -----------------------------
# Laplacian + "distinct levels"
# -----------------------------

def normalized_laplacian(A):
    """L = I - D^{-1/2} A D^{-1/2} (symmetric normalized Laplacian)."""
    d = np.array(A.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        dinv_sqrt = 1.0 / np.sqrt(d)
    dinv_sqrt[~np.isfinite(dinv_sqrt)] = 0.0
    Dinv = sparse.diags(dinv_sqrt)
    L = sparse.eye(A.shape[0], format="csr") - Dinv @ A @ Dinv
    return L

def distinct_levels(vals, tol0=1e-10, rtol=1e-6):
    vals = np.sort(np.real(vals))
    vals = vals[vals > tol0]  # drop ~0 mode(s)
    if vals.size == 0:
        raise RuntimeError("No non-zero eigenvalues.")
    levels = []
    cur = vals[0]
    mult = 1
    for v in vals[1:]:
        if abs(v - cur) <= rtol * max(1.0, abs(cur)):
            mult += 1
        else:
            levels.append((float(cur), int(mult)))
            cur = v
            mult = 1
    levels.append((float(cur), int(mult)))
    return levels

def ratio_first_two_distinct(vals, tol0=1e-10, rtol=1e-6):
    levels = distinct_levels(vals, tol0=tol0, rtol=rtol)
    if len(levels) < 2:
        raise RuntimeError("Not enough distinct levels.")
    lam1, m1 = levels[0]
    lam2, m2 = levels[1]
    return lam2 / lam1, (lam1, m1), (lam2, m2), levels[:6]


def spectral_ratio_for_alpha(X, rows, cols, dist2, alpha, k_eigs=10):
    A, sigma = make_weighted_adjacency(X, rows, cols, dist2, alpha)
    L = normalized_laplacian(A)

    # Smallest eigenvalues of normalized Laplacian are in [0, 2].
    vals, _ = eigsh(L, k=k_eigs, which="SM", tol=1e-6)
    vals = np.sort(np.real(vals))

    rho, l1, l2, head = ratio_first_two_distinct(vals)
    return rho, vals, l1, l2, head, sigma


# -----------------------------
# Main experiment: compare curves
# -----------------------------

def run_weighted_laplacian_curve(
    N_points=1500,
    k_nn=14,
    alphas=np.linspace(-2.0, 2.0, 17),
    repeats_graph=6,
    seed=0
):
    rng = np.random.default_rng(seed)

    # Fix an isotropic cloud on S^3
    X = sample_S3(N_points, rng)

    # Precompute kNN structure once (geometry fixed)
    rows, cols, dist2 = knn_graph(X, k=k_nn)

    # For each alpha, compute spectral ratio; repeat a few times with slight resampling of points
    # (optional) to get error bars. Here: we re-sample the point cloud each repeat to see robustness.
    means, errs = [], []

    print("Weighted Laplacian spectral curve on S^3 (kNN graph)\n")
    for a in alphas:
        vals = []
        for r in range(repeats_graph):
            # resample points per repeat (comment these 3 lines if you want geometry fixed across repeats)
            Xr = sample_S3(N_points, rng)
            rr, cc, d2 = knn_graph(Xr, k=k_nn)

            rho, _, l1, l2, head, sigma = spectral_ratio_for_alpha(Xr, rr, cc, d2, a, k_eigs=12)
            vals.append(rho)

        vals = np.array(vals)
        m = vals.mean()
        s = vals.std(ddof=1) if repeats_graph > 1 else 0.0
        ci = 1.96 * s / np.sqrt(max(1, repeats_graph))

        means.append(m)
        errs.append(ci)

        print(f"alpha={a:+.2f} | rho=λ2/λ1={m:.6f}  95%CI≈±{ci:.6f}")

    # Plot
    plt.figure(figsize=(9, 6))
    plt.errorbar(alphas, means, yerr=errs, fmt="o-", capsize=4, label="ρ(α) = λ2/λ1 (Laplacien pondéré)")
    plt.axhline(y=8/3, linestyle="--", color="black", label="Référence 8/3")
    plt.xlabel("Biais α")
    plt.ylabel("Ratio spectral ρ(α)")
    plt.title("Courbe spectrale sur S³ via Laplacien pondéré anisotrope")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_weighted_laplacian_curve()
