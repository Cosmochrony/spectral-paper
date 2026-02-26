"""
Test "projective entropy -> logdet(L) -> Green kernel -> 1/r -> Poisson" on a simple graph Laplacian.

Adds:
- Stochastic estimate of Delta logdet(A) with A = L + eps I (Hutchinson + log-series).
- Stochastic estimate of Tr(A0^{-1} DeltaA) to validate first-order link:
  Delta logdet(A) ~ Tr(A0^{-1} DeltaA) for small perturbations.

Notes:
- Uses a log-series: log(A) = log(alpha)I + log(I - B) with B = I - A/alpha, ||B||<1.
- alpha must be > lambda_max(A). We estimate lambda_max by eigsh.
"""

from __future__ import annotations

import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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
                        w = kappa
                        rows.append(p)
                        cols.append(q)
                        data.append(-w)
                        diag += w
                rows.append(p)
                cols.append(p)
                data.append(diag)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def apply_local_mass(
    L: sp.csr_matrix,
    nx: int,
    ny: int,
    nz: int,
    center: tuple[int, int, int],
    factor: float = 0.5,
    radius: int = 1,
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

  # We'll track which rows are affected (ball nodes + their neighbors).
  affected = set(np.where(in_ball)[0].tolist())

  # Scale couplings from nodes inside ball to their neighbors.
  for p in np.where(in_ball)[0]:
    row = L.rows[p]
    dat = L.data[p]
    for t, q in enumerate(row):
      if q != p:
        dat[t] *= factor
        affected.add(q)  # neighbor row's diagonal must be updated too

  # Recompute diagonals ONLY for affected nodes to keep row-sum zero.
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

def solve_green_column(L: sp.csr_matrix, source: int, eps: float = 1e-8) -> np.ndarray:
    n = L.shape[0]
    A = L + eps * sp.eye(n, format="csr")
    b = np.zeros(n, dtype=float)
    b[source] = 1.0
    return spla.spsolve(A, b)


def radial_profile(
    phi: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    center: tuple[int, int, int],
    rmax: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    n = L.shape[0]
    b = np.zeros(n, dtype=float)
    b[source] = 1.0
    return L @ phi - b


def fit_a_over_r(r: np.ndarray, p: np.ndarray, r_lo: int, r_hi: int) -> tuple[float, float]:
    fit_mask = (r >= r_lo) & (r <= r_hi)
    X = np.vstack([1.0 / np.maximum(r[fit_mask], 1), np.ones(np.sum(fit_mask))]).T
    a, b = np.linalg.lstsq(X, p[fit_mask], rcond=None)[0]
    return float(a), float(b)


def estimate_lambda_max(A: sp.csr_matrix, iters: int = 400) -> float:
    # Largest algebraic eigenvalue for SPD matrix A
    val = spla.eigsh(A, k=1, which="LA", return_eigenvectors=False, maxiter=iters)[0]
    return float(val)


def hutchinson_logdet(
    A: sp.csr_matrix,
    alpha: float,
    n_probe: int = 30,
    n_terms: int = 40,
    seed: int = 0,
) -> float:
    """
    Estimate logdet(A) for SPD A using:
      log(A) = log(alpha)I + log(I - B),  B = I - A/alpha.
      log(I - B) = -sum_{k>=1} B^k / k   if ||B|| < 1.
    Then: logdet(A) = Tr(log(A)) ≈ E_z[z^T log(A) z] with Rademacher probes.

    alpha must be > lambda_max(A), ideally a bit larger.
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    I = sp.eye(n, format="csr")
    B = I - (A * (1.0 / alpha))

    total = 0.0
    for _ in range(n_probe):
        z = rng.integers(0, 2, size=n, dtype=np.int8)
        z = 2.0 * z.astype(np.float64) - 1.0  # Rademacher ±1

        # Start with z^T [log(alpha)I] z = log(alpha) * ||z||^2 = log(alpha) * n
        acc = math.log(alpha) * float(n)

        # Add z^T log(I - B) z ≈ -sum (1/k) z^T B^k z
        v = z.copy()
        for k in range(1, n_terms + 1):
            v = B @ v
            acc += -(1.0 / k) * float(z @ v)

        total += acc

    return total / float(n_probe)


def hutchinson_trace_Ainv_dA(
    A0: sp.csr_matrix,
    dA: sp.csr_matrix,
    n_probe: int = 30,
    seed: int = 1,
) -> float:
    """
    Estimate Tr(A0^{-1} dA) by Hutchinson:
      Tr(M) = E[z^T M z], with z Rademacher.
      Here M = A0^{-1} dA, so z^T A0^{-1} dA z = x^T dA z with x = A0^{-1} z.
    """
    n = A0.shape[0]
    rng = np.random.default_rng(seed)

    solve = spla.factorized(A0.tocsc())

    total = 0.0
    for _ in range(n_probe):
        z = rng.integers(0, 2, size=n, dtype=np.int8)
        z = 2.0 * z.astype(np.float64) - 1.0
        x = solve(z)
        total += float(x @ (dA @ z))
    return total / float(n_probe)


def main() -> None:
    nx, ny, nz = 25, 25, 25
    center = (nx // 2, ny // 2, nz // 2)
    source = idx(*center, nx, ny)

    L0 = build_grid_laplacian(nx, ny, nz, kappa=1.0)
    Lm = apply_local_mass(L0, nx, ny, nz, center=center, factor=0.35, radius=1)

    eps = 1e-6
    A0 = L0 + eps * sp.eye(L0.shape[0], format="csr")
    Am = Lm + eps * sp.eye(Lm.shape[0], format="csr")
    dA = Am - A0

    # Green column + gauge fix for 1/r check
    phi0 = spla.spsolve(A0, sp.csr_matrix(([1.0], ([source], [0])), shape=(A0.shape[0], 1)).toarray().ravel())
    phim = spla.spsolve(Am, sp.csr_matrix(([1.0], ([source], [0])), shape=(Am.shape[0], 1)).toarray().ravel())

    phi0c = phi0 - phi0.mean()
    phimc = phim - phim.mean()
    dphic = (phim - phi0) - (phim - phi0).mean()

    rmax = 12
    r, p0 = radial_profile(phi0c, nx, ny, nz, center=center, rmax=rmax)
    _, pm = radial_profile(phimc, nx, ny, nz, center=center, rmax=rmax)
    _, dp = radial_profile(dphic, nx, ny, nz, center=center, rmax=rmax)

    r_lo, r_hi = 3, 8
    a0, b0 = fit_a_over_r(r, p0, r_lo=r_lo, r_hi=r_hi)
    am, bm = fit_a_over_r(r, pm, r_lo=r_lo, r_hi=r_hi)
    ad, bd = fit_a_over_r(r, dp, r_lo=r_lo, r_hi=r_hi)

    print(f"Far-field fit p(r) ~ a/r + b (mean removed), window r=[{r_lo},{r_hi}]")
    print(f"  baseline: a={a0:.6e}, b={b0:.6e}")
    print(f"  mass    : a={am:.6e}, b={bm:.6e}")
    print(f"  delta   : a={ad:.6e}, b={bd:.6e}")

    # Poisson residual check (L phi - delta) away from source neighborhood
    res0 = poisson_residual(L0, phi0, source)
    resm = poisson_residual(Lm, phim, source)

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

    print("\n r | p0c(r)      pmc(r)      dpc(r)      1/r")
    for rr in range(1, rmax + 1):
        inv = 1.0 / rr
        print(f"{rr:2d} | {p0[rr]: .6e} {pm[rr]: .6e} {dp[rr]: .6e} {inv: .6e}")

    # ---- logdet / trace tests ----
    lam0 = estimate_lambda_max(A0)
    lamm = estimate_lambda_max(Am)
    alpha = 1.05 * max(lam0, lamm)

    n_probe = 30
    n_terms = 40

    logdet0 = hutchinson_logdet(A0, alpha=alpha, n_probe=n_probe, n_terms=n_terms, seed=10)
    logdetm = hutchinson_logdet(Am, alpha=alpha, n_probe=n_probe, n_terms=n_terms, seed=11)
    dlogdet = logdetm - logdet0

    tr_est = hutchinson_trace_Ainv_dA(A0, dA, n_probe=n_probe, seed=12)

    print("\nStochastic logdet / trace check (A=L+eps I):")
    print(f"  lambda_max(A0) ~ {lam0:.6e}, lambda_max(Am) ~ {lamm:.6e}, alpha ~ {alpha:.6e}")
    print(f"  logdet(A0) ~ {logdet0:.6e}")
    print(f"  logdet(Am) ~ {logdetm:.6e}")
    print(f"  Delta logdet ~ {dlogdet:.6e}")
    print(f"  Tr(A0^-1 DeltaA) ~ {tr_est:.6e}")
    print(f"  ratio (Delta logdet)/(Tr) ~ {dlogdet / tr_est:.6e}")


if __name__ == "__main__":
    main()