import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


def sample_S3(N, rng):
    X = rng.normal(size=(N, 4))
    X /= np.linalg.norm(X, axis=1)[:, None]
    return X


def knn_edges(X, k=12):
    N = X.shape[0]
    tree = cKDTree(X)
    dists, idxs = tree.query(X, k=k+1)
    rows, cols, dist2 = [], [], []
    for i in range(N):
        for t in range(1, k+1):
            j = int(idxs[i, t])
            rows.append(i); cols.append(j)
            dist2.append(float(dists[i, t]**2))
    return np.array(rows), np.array(cols), np.array(dist2)


def make_isotropic_weights(dist2, sigma=None):
    eps = 1e-12
    if sigma is None:
        sigma = np.sqrt(np.median(dist2) + eps)
    w = np.exp(-dist2 / (2.0 * sigma * sigma))
    return w, sigma


def symmetrize_max(A):
    return A.maximum(A.T)


def combinatorial_laplacian(A):
    d = np.array(A.sum(axis=1)).ravel()
    D = sparse.diags(d)
    return (D - A).tocsr()


def fiber_base_energy_ratio(X, A, modes, fiber_axis=np.array([1.0,0,0,0]), n_modes_avg=6):
    """
    Same observable as before: average 8*E_parallel/E_perp over first few non-trivial eigenmodes.
    """
    A = A.tocoo()
    i = A.row
    j = A.col
    w = A.data

    d = X[j] - X[i]
    dn = np.linalg.norm(d, axis=1)
    dn[dn < 1e-12] = 1.0
    u = d / dn[:, None]

    c = (u @ fiber_axis)
    c2 = c*c
    s2 = 1.0 - c2

    ratios = []
    used = 0
    for m in range(modes.shape[1]):
        f = modes[:, m]
        if np.std(f) < 1e-10:
            continue
        df = f[j] - f[i]
        Epar = np.sum(w * c2 * df*df)
        Eper = np.sum(w * s2 * df*df)
        if Eper <= 1e-18:
            continue
        ratios.append(8.0 * Epar / Eper)
        used += 1
        if used >= n_modes_avg:
            break

    return float(np.mean(ratios)) if ratios else np.nan


def run_curve(N_points=1500, k_nn=14, alphas=np.linspace(-2,2,17), repeats=6, seed=0):
    rng = np.random.default_rng(seed)
    means, errs = [], []

    print("Spectral fiber/base curve with detailed-balance edge weights\n")

    for a in alphas:
        vals = []
        for _ in range(repeats):
            X = sample_S3(N_points, rng)
            rows, cols, dist2 = knn_edges(X, k=k_nn)
            w0, sigma = make_isotropic_weights(dist2)

            # Base adjacency
            A0 = sparse.csr_matrix((w0, (rows, cols)), shape=(N_points, N_points))
            A0.sum_duplicates(); A0.eliminate_zeros()
            A0 = symmetrize_max(A0)

            # Vertex measure mu_i = exp(alpha x0^2)
            mu = np.exp(a * (X[:, 0]**2))

            # Detailed-balance edge reweighting: A_ij = A0_ij * sqrt(mu_i mu_j)
            # Implement by scaling with s_i = sqrt(mu_i)
            s = np.sqrt(mu)
            S = sparse.diags(s)
            A = (S @ A0 @ S).tocsr()
            A.sum_duplicates(); A.eliminate_zeros()

            L = combinatorial_laplacian(A)

            # Smallest eigenpairs (include the ~0 mode)
            evals, evecs = eigsh(L, k=18, which="SM", tol=1e-6)
            idx = np.argsort(np.real(evals))
            evecs = np.real(evecs[:, idx])

            Rspec = fiber_base_energy_ratio(X, A, evecs, n_modes_avg=6)
            vals.append(Rspec)

        vals = np.array(vals, dtype=float)
        m = np.nanmean(vals)
        s = np.nanstd(vals, ddof=1)
        ci = 1.96 * s / np.sqrt(repeats)

        means.append(m); errs.append(ci)
        print(f"alpha={a:+.2f} | R_spec≈{m:.6f}  95%CI≈±{ci:.6f}")

    # plot
    plt.figure(figsize=(9,6))
    plt.errorbar(alphas, means, yerr=errs, fmt="o-", capsize=4, label="R_spec(α) (DB edges)")
    plt.axhline(y=8/3, linestyle="--", color="black", label="Référence 8/3")
    plt.xlabel("Biais α")
    plt.ylabel("Ratio spectral R_spec(α)")
    plt.title("Ratio fibre/base via Laplacien (arêtes réversibles ~ √(μ_i μ_j))")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_curve()
