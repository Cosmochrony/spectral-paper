from __future__ import annotations

import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def idx(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return i + nx * (j + ny * k)


def build_grid_laplacian(nx: int, ny: int, nz: int, kappa: float = 1.0) -> sp.csr_matrix:
    n = nx * ny * nz
    rows, cols, data = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = idx(i, j, k, nx, ny)
                diag = 0.0
                for di, dj, dk in (
                    (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1),
                ):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        q = idx(ii, jj, kk, nx, ny)
                        rows.append(p)
                        cols.append(q)
                        data.append(-kappa)
                        diag += kappa
                rows.append(p)
                cols.append(p)
                data.append(diag)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def apply_local_mass_symmetric(
    L: sp.csr_matrix,
    nx: int,
    ny: int,
    nz: int,
    center: tuple[int, int, int],
    factor: float,
    radius: int,
) -> sp.csr_matrix:
    n = nx * ny * nz
    cx, cy, cz = center

    in_ball = np.zeros(n, dtype=bool)
    for k in range(max(0, cz - radius), min(nz, cz + radius + 1)):
        for j in range(max(0, cy - radius), min(ny, cy + radius + 1)):
            for i in range(max(0, cx - radius), min(nx, cx + radius + 1)):
                if (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2 <= radius ** 2:
                    in_ball[idx(i, j, k, nx, ny)] = True

    L = L.tolil(copy=True)
    ball_nodes = set(np.where(in_ball)[0].tolist())
    affected = set(ball_nodes)

    for p in list(ball_nodes):
        row_p = L.rows[p]
        dat_p = L.data[p]
        for t, q in enumerate(row_p):
            if q == p:
                continue
            dat_p[t] *= factor
            affected.add(q)

            row_q = L.rows[q]
            dat_q = L.data[q]
            for s, qq in enumerate(row_q):
                if qq == p:
                    dat_q[s] *= factor
                    break

    for p in affected:
        row = L.rows[p]
        dat = L.data[p]
        diag_idx = None
        s_off = 0.0
        for t, q in enumerate(row):
            if q == p:
                diag_idx = t
            else:
                s_off += dat[t]
        if diag_idx is None:
            row.append(p)
            dat.append(-s_off)
        else:
            dat[diag_idx] = -s_off

    return L.tocsr()


def radial_profile(
    phi: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    center: tuple[int, int, int],
    rmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    sums = np.zeros(rmax + 1, dtype=float)
    cnts = np.zeros(rmax + 1, dtype=int)

    for k in range(nz):
        dz = k - cz
        for j in range(ny):
            dy = j - cy
            for i in range(nx):
                dx = i - cx
                r = int(math.floor(math.sqrt(dx * dx + dy * dy + dz * dz)))
                if r <= rmax:
                    p = idx(i, j, k, nx, ny)
                    sums[r] += phi[p]
                    cnts[r] += 1

    r = np.arange(rmax + 1)
    prof = np.zeros_like(sums)
    mask = cnts > 0
    prof[mask] = sums[mask] / cnts[mask]
    return r, prof


def fit_a_over_r(r: np.ndarray, p: np.ndarray, r_lo: int, r_hi: int) -> tuple[float, float]:
    fit_mask = (r >= r_lo) & (r <= r_hi)
    X = np.vstack([1.0 / np.maximum(r[fit_mask], 1), np.ones(np.sum(fit_mask))]).T
    a, b = np.linalg.lstsq(X, p[fit_mask], rcond=None)[0]
    return float(a), float(b)


def support_indices(M: sp.csr_matrix) -> np.ndarray:
    M = M.tocoo()
    S = np.unique(np.concatenate([M.row, M.col]))
    return S


def dense_submatrix(M: sp.csr_matrix, S: np.ndarray) -> np.ndarray:
    return M[S, :][:, S].toarray()


def delta_logdet_local(A0: sp.csr_matrix, dA: sp.csr_matrix) -> tuple[float, float, int]:
    S = support_indices(dA)
    m = int(S.size)
    if m == 0:
        return 0.0, 0.0, 0

    dA_SS = dense_submatrix(dA, S)
    solve = spla.factorized(A0.tocsc())

    Ainv_SS = np.zeros((m, m), dtype=float)
    for col, j in enumerate(S):
        ej = np.zeros(A0.shape[0], dtype=float)
        ej[int(j)] = 1.0
        x = solve(ej)
        Ainv_SS[:, col] = x[S]

    X = Ainv_SS @ dA_SS
    tr1 = float(np.trace(X))

    M = np.eye(m) + X
    sign, logabs = np.linalg.slogdet(M)
    if sign <= 0:
        raise ValueError(f"Non-positive determinant in local update: sign={sign}")
    return float(logabs), tr1, m


def run_scan() -> None:
    nx = ny = nz = 25
    center = (nx // 2, ny // 2, nz // 2)
    source = idx(*center, nx, ny)

    eps = 1e-6
    rmax = 12
    r_lo, r_hi = 3, 8

    # Scan grid (edit as you like)
    radii = [1, 2, 3]
    factors = [0.99, 0.97, 0.95, 0.90, 0.80, 0.70]

    L0 = build_grid_laplacian(nx, ny, nz, kappa=1.0)
    n = L0.shape[0]
    A0 = L0 + eps * sp.eye(n, format="csr")

    b = np.zeros(n, dtype=float)
    b[source] = 1.0

    phi0 = spla.spsolve(A0, b)
    phi0c = phi0 - phi0.mean()
    r, p0 = radial_profile(phi0c, nx, ny, nz, center=center, rmax=rmax)
    a0, b0 = fit_a_over_r(r, p0, r_lo=r_lo, r_hi=r_hi)

    rows = []
    for radius in radii:
        for factor in factors:
            Lm = apply_local_mass_symmetric(L0, nx, ny, nz, center=center, factor=factor, radius=radius)
            Am = Lm + eps * sp.eye(n, format="csr")
            dA = Am - A0

            phim = spla.spsolve(Am, b)
            phimc = phim - phim.mean()

            _, pm = radial_profile(phimc, nx, ny, nz, center=center, rmax=rmax)
            am, bm = fit_a_over_r(r, pm, r_lo=r_lo, r_hi=r_hi)

            dlogdet_loc, tr1_loc, m = delta_logdet_local(A0, dA)

            rows.append({
                "radius": radius,
                "factor": factor,
                "support_size": m,
                "a0": a0,
                "am": am,
                "delta_a": am - a0,
                "b0": b0,
                "bm": bm,
                "delta_logdet_local": dlogdet_loc,
                "trace1_local": tr1_loc,
                "ratio_logdet_over_trace": dlogdet_loc / tr1_loc if tr1_loc != 0 else np.nan,
            })

            print(f"done radius={radius} factor={factor} delta_a={am - a0:+.3e} dlogdet={dlogdet_loc:+.3e}")

    df = pd.DataFrame(rows).sort_values(["radius", "factor"], ascending=[True, False])
    print("\nScan table:")
    print(df.to_string(index=False))

    df.to_csv("scan_deltaa_logdet.csv", index=False)

    # Figure 1: scatter delta_a vs delta_logdet_local
    plt.figure()
    plt.xlabel(r"$\Delta \log \det_{\mathrm{local}}(A)$")
    plt.ylabel(r"$\Delta a$ from $a/r$ fit")
    plt.scatter(df["delta_logdet_local"].values, df["delta_a"].values, s=25)
    plt.tight_layout()
    plt.savefig("scan_scatter_deltaa_vs_dlogdet.png", dpi=200)

    # Figure 2: heatmap-like image of delta_a over (radius,factor)
    # We'll build a matrix with rows=radii, cols=factors (sorted descending as above).
    fac_sorted = sorted(factors, reverse=True)
    rad_sorted = sorted(radii)
    mat = np.full((len(rad_sorted), len(fac_sorted)), np.nan, dtype=float)
    for i, rad in enumerate(rad_sorted):
        for j, fac in enumerate(fac_sorted):
            sub = df[(df["radius"] == rad) & (df["factor"] == fac)]
            if len(sub) == 1:
                mat[i, j] = float(sub["delta_a"].iloc[0])

    plt.figure()
    plt.xlabel("factor")
    plt.ylabel("radius")
    plt.xticks(range(len(fac_sorted)), [str(f) for f in fac_sorted])
    plt.yticks(range(len(rad_sorted)), [str(r) for r in rad_sorted])
    plt.imshow(mat, aspect="auto", origin="lower")
    plt.colorbar(label=r"$\Delta a$")
    plt.tight_layout()
    plt.savefig("scan_heatmap_deltaa.png", dpi=200)

    print("\nWrote:")
    print("  scan_deltaa_logdet.csv")
    print("  scan_scatter_deltaa_vs_dlogdet.png")
    print("  scan_heatmap_deltaa.png")


if __name__ == "__main__":
    run_scan()