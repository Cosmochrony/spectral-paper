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


def first_modes_generalized(L, M_diag, k_eigs=12):
    """
    Solve L f = λ M f for smallest eigenvalues (excluding the ~0 mode later).
    L: symmetric PSD
    M_diag: positive diagonal of M
    """
    # Convert generalized to standard by M^{-1/2} L M^{-1/2}
    eps = 1e-12
    Minv_sqrt = 1.0 / np.sqrt(np.maximum(M_diag, eps))
    S = sparse.diags(Minv_sqrt)
    Astd = (S @ L @ S).tocsr()
    vals, vecs = eigsh(Astd, k=k_eigs, which="SM", tol=1e-6)
    vals = np.real(vals)
    vecs = np.real(vecs)

    # back-transform eigenvectors to f = M^{-1/2} u
    f = S @ vecs
    return np.sort(vals), f


def fiber_base_energy_ratio(X, A, f_modes, alpha, fiber_axis=np.array([1.0,0,0,0]), n_modes_avg=6):
    """
    Compute a spectral analogue of R(α) by decomposing edge directions into
    components parallel vs transverse to the fiber axis, and averaging energies
    over the first few non-trivial eigenmodes.
    """
    # edges (i<->j) from adjacency
    A = A.tocoo()
    i = A.row
    j = A.col
    w = A.data

    # edge unit directions in ambient R^4
    d = X[j] - X[i]
    dn = np.linalg.norm(d, axis=1)
    dn[dn < 1e-12] = 1.0
    u = d / dn[:, None]

    # parallel component squared relative to fiber axis
    c = (u @ fiber_axis)
    c2 = c*c
    s2 = 1.0 - c2

    # take first n_modes_avg non-zero-ish modes (skip constant by variance)
    # We'll select modes with non-trivial variation.
    energies = []
    count = 0
    for m in range(f_modes.shape[1]):
        f = f_modes[:, m]
        if np.std(f) < 1e-10:
            continue
        # edge differences
        df = f[j] - f[i]
        Epar = np.sum(w * c2 * df*df)
        Eper = np.sum(w * s2 * df*df)
        if Eper <= 1e-18:
            continue
        energies.append(8.0 * Epar / Eper)
        count += 1
        if count >= n_modes_avg:
            break

    if not energies:
        return np.nan
    return float(np.mean(energies))


def run_curve(N_points=1500, k_nn=14, alphas=np.linspace(-2,2,17), repeats=6, seed=0):
    rng = np.random.default_rng(seed)

    means, errs = [], []

    print("Spectral fiber/base curve via Laplacian with vertex measure\n")

    for a in alphas:
        vals = []
        for _ in range(repeats):
            X = sample_S3(N_points, rng)
            rows, cols, dist2 = knn_edges(X, k=k_nn)
            w0, sigma = make_isotropic_weights(dist2)

            A = sparse.csr_matrix((w0, (rows, cols)), shape=(N_points, N_points))
            A.sum_duplicates(); A.eliminate_zeros()
            A = symmetrize_max(A)

            L = combinatorial_laplacian(A)

            # vertex measure μ_i(α) = exp(α x0^2)
            mu = np.exp(a * (X[:, 0]**2))
            M = mu  # diagonal

            evals, fmodes = first_modes_generalized(L, M, k_eigs=18)
            Rspec = fiber_base_energy_ratio(X, A, fmodes, a, n_modes_avg=6)
            vals.append(Rspec)

        vals = np.array(vals, dtype=float)
        m = np.nanmean(vals)
        s = np.nanstd(vals, ddof=1)
        ci = 1.96 * s / np.sqrt(repeats)

        means.append(m); errs.append(ci)
        print(f"alpha={a:+.2f} | R_spec≈{m:.6f}  95%CI≈±{ci:.6f}")

    # plot
    plt.figure(figsize=(9,6))
    plt.errorbar(alphas, means, yerr=errs, fmt="o-", capsize=4, label="R_spec(α) (énergie fibre/base)")
    plt.axhline(y=8/3, linestyle="--", color="black", label="Référence 8/3")
    plt.xlabel("Biais α")
    plt.ylabel("Ratio spectral R_spec(α)")
    plt.title("Ratio fibre/base via Laplacien pondéré par mesure (S³, kNN)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_curve()
