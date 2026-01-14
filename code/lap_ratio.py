#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check whether lambda2/lambda1 ≈ 8/3 emerges naturally from the first non-zero
eigenmodes of the graph 0-form Laplacian Δ_G^(0).

Δ_G^(0) is the (combinatorial) graph Laplacian L = D - A by default.
Optionally use normalized Laplacian.

Requires: numpy, scipy
Optional (for graph generation): networkx
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


# -----------------------------
# Graph builders (sparse)
# -----------------------------

def grid_2d(n: int, m: Optional[int] = None, periodic: bool = False) -> sparse.csr_matrix:
    """4-neighbor 2D grid adjacency as CSR."""
    if m is None:
        m = n
    N = n * m

    rows = []
    cols = []

    def idx(i, j): return i * m + j

    for i in range(n):
        for j in range(m):
            u = idx(i, j)
            # right neighbor
            if j + 1 < m:
                v = idx(i, j + 1)
                rows += [u, v]
                cols += [v, u]
            elif periodic and m > 1:
                v = idx(i, 0)
                rows += [u, v]
                cols += [v, u]
            # down neighbor
            if i + 1 < n:
                v = idx(i + 1, j)
                rows += [u, v]
                cols += [v, u]
            elif periodic and n > 1:
                v = idx(0, j)
                rows += [u, v]
                cols += [v, u]

    data = np.ones(len(rows), dtype=np.float64)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    A.sum_duplicates()
    A.eliminate_zeros()
    return A


def erdos_renyi(n: int, p: float, seed: int = 0) -> sparse.csr_matrix:
    """Sparse ER adjacency (undirected, no self-loops)."""
    rng = np.random.default_rng(seed)
    # Upper triangle mask
    triu = sparse.random(n, n, density=p/2, format="coo", random_state=rng,
                         data_rvs=lambda k: np.ones(k))
    triu = sparse.triu(triu, k=1)
    A = triu + triu.T
    A = A.tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    return A


def watts_strogatz(n: int, k: int, beta: float, seed: int = 0) -> sparse.csr_matrix:
    """
    Watts-Strogatz via networkx if available.
    k must be even and < n.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise SystemExit("watts_strogatz requires networkx: pip install networkx") from e

    if k % 2 != 0:
        raise ValueError("k must be even for Watts–Strogatz.")
    G = nx.watts_strogatz_graph(n, k, beta, seed=seed)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)
    return A


# -----------------------------
# Laplacian + spectrum
# -----------------------------

@dataclass
class SpectrumResult:
    lambdas: np.ndarray        # sorted eigenvalues (starting at 0)
    ratio_21: float            # lambda2_nonzero / lambda1_nonzero
    lambda1: float             # first non-zero
    lambda2: float             # second non-zero
    gap01: float               # lambda1_nonzero - lambda0 (lambda0~0)
    info: Dict[str, float]


def laplacian_spectrum(
    A: sparse.csr_matrix,
    k_eigs: int = 6,
    normalized: bool = False,
    tol: float = 1e-10,
) -> SpectrumResult:
    """
    Compute smallest k_eigs eigenvalues of graph Laplacian (0-form).
    For connected graphs, eigenvalue 0 has multiplicity 1.
    """
    if A.shape[0] < k_eigs + 1:
        k_eigs = max(2, A.shape[0] - 1)

    if normalized:
        L = csgraph.laplacian(A, normed=True)
    else:
        L = csgraph.laplacian(A, normed=False)

    # Compute smallest eigenvalues. 'SM' = smallest magnitude.
    # For Laplacian (PSD), it's fine.
    vals, _ = eigsh(L, k=k_eigs, which="SM", tol=1e-8)
    vals = np.sort(np.real(vals))

    ratio_d, (lam1, m1), (lam2, m2), head = ratio_first_two_distinct(vals, tol0=tol, rtol=1e-6)

    print("First distinct non-zero levels (value, multiplicity):", head)
    print(f"λ1(distinct)={lam1:.6e} (mult={m1})  λ2(distinct)={lam2:.6e} (mult={m2})")
    print(f"λ2/λ1 (distinct-level ratio) = {ratio_d:.12f}")

    # Identify first two non-zero eigenvalues robustly
    nonzero = vals[vals > tol]
    if len(nonzero) < 2:
        raise RuntimeError("Graph may be disconnected or too small: not enough non-zero eigenvalues.")

    lambda1 = float(nonzero[0])
    lambda2 = float(nonzero[1])
    ratio_21 = lambda2 / lambda1

    return SpectrumResult(
        lambdas=vals,
        ratio_21=ratio_21,
        lambda1=lambda1,
        lambda2=lambda2,
        gap01=lambda1 - float(vals[0]),
        info={
            "n": float(A.shape[0]),
            "nnz": float(A.nnz),
            "avg_degree": float(A.nnz / A.shape[0]),
            "normalized": float(1 if normalized else 0),
        },
    )


# -----------------------------
# Experiment harness
# -----------------------------

def run_one(kind: str, **kwargs) -> SpectrumResult:
    if kind == "grid2d":
        A = grid_2d(kwargs["n"], kwargs.get("m"), periodic=kwargs.get("periodic", False))
    elif kind == "er":
        A = erdos_renyi(kwargs["n"], kwargs["p"], seed=kwargs.get("seed", 0))
    elif kind == "ws":
        A = watts_strogatz(kwargs["n"], kwargs["k"], kwargs["beta"], seed=kwargs.get("seed", 0))
    else:
        raise ValueError(f"Unknown graph kind: {kind}")

    return laplacian_spectrum(
        A,
        k_eigs=kwargs.get("k_eigs", 8),
        normalized=kwargs.get("normalized", False),
        tol=kwargs.get("tol", 1e-10),
    )


def sweep_grid_sizes(
    sizes: List[int],
    periodic: bool,
    normalized: bool,
    k_eigs: int,
) -> None:
    target = 8.0 / 3.0
    print(f"Target ratio 8/3 = {target:.12f}\n")

    for n in sizes:
        res = run_one(
            "grid2d",
            n=n,
            m=n,
            periodic=periodic,
            normalized=normalized,
            k_eigs=k_eigs,
        )
        err = abs(res.ratio_21 - target) / target
        print(
            f"grid {n}x{n} | λ1={res.lambda1:.6e}  λ2={res.lambda2:.6e}  "
            f"λ2/λ1={res.ratio_21:.9f}  rel.err={err:.3e}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["grid2d", "er", "ws"], default="grid2d")
    ap.add_argument("--n", type=int, default=30, help="nodes per side for grid2d, or number of nodes otherwise")
    ap.add_argument("--m", type=int, default=None, help="grid2d second dimension (defaults to n)")
    ap.add_argument("--periodic", action="store_true", help="periodic boundaries for grid2d")
    ap.add_argument("--normalized", action="store_true", help="use normalized Laplacian")
    ap.add_argument("--k_eigs", type=int, default=10, help="number of smallest eigenvalues to compute")
    ap.add_argument("--tol", type=float, default=1e-10)

    # ER params
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)

    # WS params
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--beta", type=float, default=0.2)

    # sweep helper for grid
    ap.add_argument("--sweep", action="store_true", help="sweep grid sizes n=10..80")
    args = ap.parse_args()

    if args.sweep:
        sizes = list(range(10, 81, 10))
        sweep_grid_sizes(
            sizes=sizes,
            periodic=args.periodic,
            normalized=args.normalized,
            k_eigs=args.k_eigs,
        )
        return

    res = run_one(
        args.kind,
        n=args.n,
        m=args.m,
        periodic=args.periodic,
        p=args.p,
        seed=args.seed,
        k=args.k,
        beta=args.beta,
        normalized=args.normalized,
        k_eigs=args.k_eigs,
        tol=args.tol,
    )

    target = 8.0 / 3.0
    rel_err = abs(res.ratio_21 - target) / target

    print(f"Graph: {args.kind}, n={args.n}, normalized={args.normalized}")
    print(f"avg_degree ≈ {res.info['avg_degree']:.3f}  nnz={int(res.info['nnz'])}")
    print(f"Smallest eigenvalues: {res.lambdas[:min(len(res.lambdas),10)]}")
    print(f"λ1 (1st non-zero) = {res.lambda1:.12e}")
    print(f"λ2 (2nd non-zero) = {res.lambda2:.12e}")
    print(f"λ2/λ1 = {res.ratio_21:.12f}")
    print(f"Target 8/3 = {target:.12f}  (relative error {rel_err:.3e})")


def distinct_levels(eigs: np.ndarray, tol0: float = 1e-10, rtol: float = 1e-6):
  """
  Group eigenvalues into distinct 'levels' (bands) using a relative tolerance.
  Returns list of (level_value, multiplicity).
  """
  eigs = np.sort(np.real(eigs))
  eigs = eigs[eigs > tol0]  # drop ~0 modes (connected graph => one)
  if eigs.size == 0:
    raise RuntimeError("No non-zero eigenvalues found (graph disconnected / too small).")

  levels = []
  cur = eigs[0]
  mult = 1
  for v in eigs[1:]:
    # group if close to current level (relative)
    if abs(v - cur) <= rtol * max(1.0, abs(cur)):
      mult += 1
    else:
      levels.append((cur, mult))
      cur = v
      mult = 1
  levels.append((cur, mult))
  return levels


def ratio_first_two_distinct(eigs: np.ndarray, tol0: float = 1e-10, rtol: float = 1e-6):
  levels = distinct_levels(eigs, tol0=tol0, rtol=rtol)
  if len(levels) < 2:
    raise RuntimeError("Not enough distinct non-zero levels to form a ratio.")
  lam1, m1 = levels[0]
  lam2, m2 = levels[1]
  return (lam2 / lam1), (lam1, m1), (lam2, m2), levels[:6]  # show first few levels


if __name__ == "__main__":
    main()
