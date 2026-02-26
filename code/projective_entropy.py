"""
Test "projective entropy -> logdet(L) -> Green kernel -> 1/r -> Poisson" on a simple graph Laplacian.

Idea:
- Build a 3D grid graph Laplacian L (Dirichlet boundary).
- Add a localized "mass" by reducing conductivities (edge weights) around a site.
- Solve (L + eps I) phi = delta_source to get the Green column (potential).
- Check:
  (1) radial average phi(r) ~ const / r (in 3D, far from source),
  (2) discrete Poisson: L phi ~ delta_source away from regularization.
Optional:
- Compare logdet(L) before/after via stochastic trace log (Hutchinson + Lanczos-ish approximation).
  (This part is optional and can be slow; kept minimal.)
"""

from __future__ import annotations

import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def idx(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return i + nx * (j + ny * k)


def build_grid_laplacian(nx: int, ny: int, nz: int, kappa: float = 1.0) -> sp.csr_matrix:
    """
    3D 6-neighbor grid Laplacian with Dirichlet boundary (implicit).
    Weighted edges with uniform kappa.
    L = -div(kappa grad) discretized on nodes.
    """
    n = nx * ny * nz
    rows, cols, data = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = idx(i, j, k, nx, ny)
                diag = 0.0
                for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                  (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        q = idx(ii, jj, kk, nx, ny)
                        w = kappa
                        rows.append(p); cols.append(q); data.append(-w)
                        diag += w
                rows.append(p); cols.append(p); data.append(diag)
    L = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return L


def apply_local_mass(L: sp.csr_matrix, nx: int, ny: int, nz: int,
                     center: tuple[int, int, int], factor: float = 0.5,
                     radius: int = 1) -> sp.csr_matrix:
    """
    Reduce conductivities (edge weights) in a local ball around 'center' by 'factor' in (0, 1].
    Operationally: scale off-diagonal couplings for edges incident to nodes in the ball, then fix diag.
    This models a local inhibition of relaxation (lower kappa).
    """
    n = nx * ny * nz
    cx, cy, cz = center

    in_ball = np.zeros(n, dtype=bool)
    for k in range(max(0, cz - radius), min(nz, cz + radius + 1)):
        for j in range(max(0, cy - radius), min(ny, cy + radius + 1)):
            for i in range(max(0, cx - radius), min(nx, cx + radius + 1)):
                if (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2 <= radius ** 2:
                    in_ball[idx(i, j, k, nx, ny)] = True

    L = L.tolil(copy=True)

    # Scale couplings from nodes inside ball to their neighbors.
    for p in np.where(in_ball)[0]:
        row = L.rows[p]
        dat = L.data[p]
        for t, q in enumerate(row):
            if q != p:
                dat[t] *= factor

    # Recompute diagonals to keep "sum of row = 0" (Dirichlet boundary already handled by missing edges).
    for p in range(n):
        row = L.rows[p]
        dat = L.data[p]
        diag_idx = None
        s = 0.0
        for t, q in enumerate(row):
            if q == p:
                diag_idx = t
            else:
                s += dat[t]
        if diag_idx is None:
            row.append(p); dat.append(-s)
        else:
            dat[diag_idx] = -s

    return L.tocsr()


def solve_green_column(L: sp.csr_matrix, source: int, eps: float = 1e-8) -> np.ndarray:
    """
    Solve (L + eps I) phi = e_source. eps removes the zero-mode (or near-zero) and stabilizes solve.
    """
    n = L.shape[0]
    A = L + eps * sp.eye(n, format="csr")
    b = np.zeros(n, dtype=float)
    b[source] = 1.0
    phi = spla.spsolve(A, b)
    return phi


def radial_profile(phi: np.ndarray, nx: int, ny: int, nz: int,
                   center: tuple[int, int, int], rmax: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Radial average of phi over integer shells r = floor(|x-center|).
    """
    cx, cy, cz = center
    if rmax is None:
        rmax = int(math.sqrt((nx - 1) ** 2 + (ny - 1) ** 2 + (nz - 1) ** 2))
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


def poisson_residual(L: sp.csr_matrix, phi: np.ndarray, source: int) -> np.ndarray:
    """
    Compute residual: (L phi) - e_source.
    """
    n = L.shape[0]
    b = np.zeros(n, dtype=float)
    b[source] = 1.0
    return L @ phi - b


def main() -> None:
    nx, ny, nz = 25, 25, 25
    center = (nx // 2, ny // 2, nz // 2)
    source = idx(*center, nx, ny)

    L0 = build_grid_laplacian(nx, ny, nz, kappa=1.0)
    Lm = apply_local_mass(L0, nx, ny, nz, center=center, factor=0.35, radius=1)

    eps = 1e-6
    phi0 = solve_green_column(L0, source, eps=eps)
    phim = solve_green_column(Lm, source, eps=eps)

    r, p0 = radial_profile(phi0, nx, ny, nz, center=center, rmax=12)
    _, pm = radial_profile(phim, nx, ny, nz, center=center, rmax=12)

    # Fit far field to a/r + b (avoid r=0,1,2).
    fit_mask = (r >= 4) & (r <= 12)
    X = np.vstack([1.0 / np.maximum(r[fit_mask], 1), np.ones(np.sum(fit_mask))]).T
    a0, b0 = np.linalg.lstsq(X, p0[fit_mask], rcond=None)[0]
    am, bm = np.linalg.lstsq(X, pm[fit_mask], rcond=None)[0]

    print("Far-field fit p(r) ~ a/r + b")
    print(f"  baseline: a={a0:.6e}, b={b0:.6e}")
    print(f"  mass    : a={am:.6e}, b={bm:.6e}")
    print(f"  delta a : {am - a0:.6e} (strength change due to local inhibition)")

    # Check discrete Poisson residual away from the source neighborhood.
    res0 = poisson_residual(L0, phi0, source)
    resm = poisson_residual(Lm, phim, source)

    # Ignore a small ball around source where eps regularization dominates.
    ignore = set()
    cx, cy, cz = center
    for k in range(cz - 1, cz + 2):
        for j in range(cy - 1, cy + 2):
            for i in range(cx - 1, cx + 2):
                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                    ignore.add(idx(i, j, k, nx, ny))

    mask = np.ones(nx * ny * nz, dtype=bool)
    for p in ignore:
        mask[p] = False

    print("Poisson check (RMS of L phi - delta, excluding tiny ball):")
    print(f"  baseline RMS: {np.sqrt(np.mean(res0[mask] ** 2)):.6e}")
    print(f"  mass     RMS: {np.sqrt(np.mean(resm[mask] ** 2)):.6e}")

    # Quick look at delta potential (how mass changes the Green column).
    dphi = phim - phi0
    _, dp = radial_profile(dphi, nx, ny, nz, center=center, rmax=12)
    ad, bd = np.linalg.lstsq(X, dp[fit_mask], rcond=None)[0]
    print(f"Delta potential fit dp(r) ~ ad/r + bd: ad={ad:.6e}, bd={bd:.6e}")

    # Print a small table.
    print("\n r | p0(r)       pm(r)       dp(r)       1/r")
    for rr in range(1, 13):
        inv = 1.0 / rr
        print(f"{rr:2d} | {p0[rr]: .6e} {pm[rr]: .6e} {dp[rr]: .6e} {inv: .6e}")


if __name__ == "__main__":
    main()