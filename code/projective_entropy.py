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
                        rows.append(p)
                        cols.append(q)
                        data.append(-kappa)
                        diag += kappa
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
    factor: float = 0.9,
    radius: int = 1,
) -> sp.csr_matrix:
  """
  Symmetric local inhibition:
  - For every edge (p,q) with p in ball or q in ball, scale BOTH L[p,q] and L[q,p] by factor.
  - Update diagonals only for affected nodes to keep row-sum zero.
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
  ball_nodes = set(np.where(in_ball)[0].tolist())
  affected = set(ball_nodes)

  # For each ball node p, scale all couplings (p,q) and (q,p).
  for p in list(ball_nodes):
    row_p = L.rows[p]
    dat_p = L.data[p]
    for t, q in enumerate(row_p):
      if q == p:
        continue
      # scale p->q
      dat_p[t] *= factor
      affected.add(q)

      # scale q->p (find entry in row q)
      row_q = L.rows[q]
      dat_q = L.data[q]
      for s, qq in enumerate(row_q):
        if qq == p:
          dat_q[s] *= factor
          break

  # Recompute diagonals only for affected nodes to keep row-sum zero.
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


def poisson_residual(L: sp.csr_matrix, phi: np.ndarray, source: int) -> np.ndarray:
    b = np.zeros(L.shape[0], dtype=float)
    b[source] = 1.0
    return L @ phi - b


def fit_a_over_r(r: np.ndarray, p: np.ndarray, r_lo: int, r_hi: int) -> tuple[float, float]:
    fit_mask = (r >= r_lo) & (r <= r_hi)
    X = np.vstack([1.0 / np.maximum(r[fit_mask], 1), np.ones(np.sum(fit_mask))]).T
    a, b = np.linalg.lstsq(X, p[fit_mask], rcond=None)[0]
    return float(a), float(b)


def estimate_lambda_max(A: sp.csr_matrix, iters: int = 400) -> float:
    val = spla.eigsh(A, k=1, which="LA", return_eigenvectors=False, maxiter=iters)[0]
    return float(val)


def hutchinson_logdet(
    A: sp.csr_matrix,
    alpha: float,
    n_probe: int = 30,
    n_terms: int = 40,
    seed: int = 0,
) -> float:
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    I = sp.eye(n, format="csr")
    B = I - (A * (1.0 / alpha))

    total = 0.0
    for _ in range(n_probe):
        z = rng.integers(0, 2, size=n, dtype=np.int8)
        z = 2.0 * z.astype(np.float64) - 1.0

        acc = math.log(alpha) * float(n)

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

def main() -> None:
    nx, ny, nz = 25, 25, 25
    center = (nx // 2, ny // 2, nz // 2)
    source = idx(*center, nx, ny)

    L0 = build_grid_laplacian(nx, ny, nz, kappa=1.0)

    # For the trace/logdet first-order check, keep the perturbation mild.
    Lm = apply_local_mass(L0, nx, ny, nz, center=center, factor=0.9, radius=1)

    eps = 1e-6
    n = L0.shape[0]
    A0 = L0 + eps * sp.eye(n, format="csr")
    Am = Lm + eps * sp.eye(n, format="csr")
    dA = Am - A0

    # Diagnostics on dA locality
    diag_sum = float(dA.diagonal().sum())
    nnz = int(dA.nnz)
    fro = float(np.sqrt(dA.power(2).sum()))
    print("dA diagnostics:")
    print(f"  nnz(dA) = {nnz}")
    print(f"  sum diag(dA) = {diag_sum:.6e}")
    print(f"  Frobenius(dA) = {fro:.6e}")

    # Green columns
    b = np.zeros(n, dtype=float)
    b[source] = 1.0
    phi0 = spla.spsolve(A0, b)
    phim = spla.spsolve(Am, b)

    # Mean removal for the 1/r visualization (gauge fixing)
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

    print(f"\nFar-field fit p(r) ~ a/r + b (mean removed), window r=[{r_lo},{r_hi}]")
    print(f"  baseline: a={a0:.6e}, b={b0:.6e}")
    print(f"  mass    : a={am:.6e}, b={bm:.6e}")
    print(f"  delta   : a={ad:.6e}, b={bd:.6e}")

    # Poisson check
    res0 = poisson_residual(L0, phi0, source)
    resm = poisson_residual(Lm, phim, source)

    ignore = set()
    cx, cy, cz = center
    for k in range(cz - 1, cz + 2):
        for j in range(cy - 1, cy + 2):
            for i in range(cx - 1, cx + 2):
                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                    ignore.add(idx(i, j, k, nx, ny))

    mask = np.ones(n, dtype=bool)
    for p in ignore:
        mask[p] = False

    print("\nPoisson check (RMS of L phi - delta, excluding tiny ball):")
    print(f"  baseline RMS: {np.sqrt(np.mean(res0[mask] ** 2)):.6e}")
    print(f"  mass     RMS: {np.sqrt(np.mean(resm[mask] ** 2)):.6e}")

    print("\n r | p0c(r)      pmc(r)      dpc(r)      1/r")
    for rr in range(1, rmax + 1):
        inv = 1.0 / rr
        print(f"{rr:2d} | {p0[rr]: .6e} {pm[rr]: .6e} {dp[rr]: .6e} {inv: .6e}")

    # logdet / trace check
    dlogdet_loc, tr1_loc, m = delta_logdet_local(A0, dA)
    print("\nLocal check on support S:")
    print(f"  |S| = {m}")
    print(f"  Delta logdet(local) = {dlogdet_loc:.6e}")
    print(f"  Tr((A0^-1)_SS dA_SS) = {tr1_loc:.6e}")
    print(f"  ratio = {dlogdet_loc / tr1_loc:.6e}")

if __name__ == "__main__":
    main()