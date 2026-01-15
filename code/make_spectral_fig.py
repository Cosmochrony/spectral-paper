#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt


@dataclass
class Mode:
  n: int
  lam: float
  ipr: float


@dataclass
class RunResult:
  sep: int
  kappa1: float
  kappa2: float
  modes: List[Mode]
  converged: bool
  energy: Optional[float]


MODE_RE = re.compile(
  r"^\s*n=\s*(\d+)\s+lambda=\s*([0-9.eE+-]+)\s+IPR=\s*([0-9.eE+-]+)\s*$"
)
ENERGY_RE = re.compile(r"^\[relax\].*E=([0-9.eE+-]+)")
CONV_RE = re.compile(r"^\[relax\]\s+Converged\b")

STYLES = {
  0: dict(linestyle="-", marker="o", linewidth=2.2),  # λ0
  1: dict(linestyle="--", marker="o", linewidth=2.2),  # λ1 (même marqueur, autre trait)
  2: dict(linestyle="-.", marker="s", linewidth=2.0),  # λ2
  3: dict(linestyle=":", marker="s", linewidth=2.0),  # λ3
}


def run_one(
    toy_path: str,
    sep: int,
    kappa1: float,
    kappa2: float,
    steps: int = 20000,
    lr: float = 0.02,
    k_eigs: int = 6,
) -> RunResult:
  """
  Runs the toy simulation once and parses lowest eigenvalues/IPR.
  """
  cmd = [
    sys.executable,
    toy_path,
    "--sep", str(sep),
    "--pin",
    "--pin_mode", "asym",
    "--kappa_pin", str(kappa1),
    "--kappa_pin2", str(kappa2),
    "--steps", str(steps),
    "--lr", str(lr),
    "--k_eigs", str(k_eigs),
    "--no_plots",
  ]

  proc = subprocess.run(cmd, capture_output=True, text=True)
  out = proc.stdout.splitlines()

  modes: List[Mode] = []
  converged = False
  last_energy: Optional[float] = None

  for line in out:
    m = MODE_RE.match(line)
    if m:
      modes.append(Mode(n=int(m.group(1)), lam=float(m.group(2)), ipr=float(m.group(3))))
      continue
    if CONV_RE.match(line):
      converged = True
      continue
    em = ENERGY_RE.match(line)
    if em:
      last_energy = float(em.group(1))

  if proc.returncode != 0:
    raise RuntimeError(
      f"Command failed (code={proc.returncode}).\n"
      f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

  return RunResult(sep=sep, kappa1=kappa1, kappa2=kappa2, modes=modes, converged=converged, energy=last_energy)


def extract_series(results: List[RunResult], mode_indices: List[int]) -> Dict[
  int, Tuple[List[int], List[float], List[float]]]:
  """
  For each mode index n, returns (seps, lambdas, iprs) over runs.
  Missing modes are skipped.
  """
  series: Dict[int, Tuple[List[int], List[float], List[float]]] = {}
  for n in mode_indices:
    seps: List[int] = []
    lams: List[float] = []
    iprs: List[float] = []
    for r in results:
      found = next((m for m in r.modes if m.n == n), None)
      if found is None:
        continue
      seps.append(r.sep)
      lams.append(found.lam)
      iprs.append(found.ipr)
    series[n] = (seps, lams, iprs)
  return series


def plot_sep_scan(results_sym: List[RunResult], outfile: str = "spectral_scan_sep.pdf", mode_indices=(0, 1, 2, 3)):
  """
  Two-panel plot: lambdas vs sep, and IPR vs sep, for selected eigenmodes.
  """
  # Sanity: report convergence
  bad = [r for r in results_sym if not r.converged]
  if bad:
    print("WARNING: Some runs did not converge (still plotted):", [r.sep for r in bad])

  series = extract_series(results_sym, list(mode_indices))

  fig = plt.figure(figsize=(10, 6))
  gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.25)

  ax1 = fig.add_subplot(gs[0, 0])
  ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

  # Panel (a): eigenvalues
  for n, (seps, lams, iprs) in series.items():
    style = STYLES.get(n, {})
    ax1.plot(
      seps, lams,
      label=f"$\\lambda_{n}$",
      **style
    )
  ax1.set_ylabel("Lowest eigenvalues $\\lambda_n$")
  ax1.set_yscale("log")  # useful since you saw ~1e-2 up to ~2
  ax1.grid(True, which="both", linewidth=0.5)
  ax1.legend(frameon=False, fontsize=9, ncol=2)

  # Panel (b): IPR
  for n, (seps, lams, iprs) in series.items():
    style = STYLES.get(n, {})
    ax2.plot(
      seps, iprs,
      label=f"IPR($n={n}$)",
      **style
    )
  ax2.set_xlabel("Separation $\\,\\mathrm{sep}$")
  ax2.set_ylabel("IPR")
  ax2.grid(True, which="both", linewidth=0.5)

  # Title
  k1 = results_sym[0].kappa1 if results_sym else None
  k2 = results_sym[0].kappa2 if results_sym else None
  fig.suptitle(f"Pinned kink–antikink stability spectrum vs separation (kappa1={k1}, kappa2={k2})", fontsize=12)

  fig.savefig(outfile, bbox_inches="tight")
  print(f"Wrote: {outfile}")


def plot_kappa_split(
    toy_path: str,
    sep: int,
    kappa1: float,
    kappa2_list: List[float],
    steps: int = 80000,
    lr: float = 0.02,
    k_eigs: int = 6,
    outfile: str = "spectral_split_kappa2.pdf",
    mode_indices=(0, 1, 2, 3),
):
  """
  Optional plot: lambdas vs kappa2 at fixed sep (shows lifting of degeneracy).
  """
  results: List[RunResult] = []
  for k2 in kappa2_list:
    print(f"Running sep={sep}, kappa1={kappa1}, kappa2={k2} ...")
    results.append(run_one(toy_path, sep=sep, kappa1=kappa1, kappa2=k2, steps=steps, lr=lr, k_eigs=k_eigs))

  # Build series vs kappa2
  fig, ax = plt.subplots(figsize=(10, 4))
  for n in mode_indices:
    xs: List[float] = []
    ys: List[float] = []
    for r in results:
      found = next((m for m in r.modes if m.n == n), None)
      if found:
        xs.append(r.kappa2)
        ys.append(found.lam)
    ax.plot(xs, ys, marker="o", linestyle="-", label=f"$\\lambda_{n}$")

  ax.set_xlabel("Secondary pin strength $\\kappa_2$")
  ax.set_ylabel("Eigenvalue $\\lambda_n$")
  ax.set_yscale("log")
  ax.grid(True, which="both", linewidth=0.5)
  ax.legend(ncol=2, fontsize=10)
  ax.set_title(f"Degeneracy lifting vs $\\kappa_2$ at sep={sep} (kappa1={kappa1})")

  fig.savefig(outfile, bbox_inches="tight")
  print(f"Wrote: {outfile}")


if __name__ == "__main__":
  # ---- CONFIG ----
  TOY = "../scripts/toy_cosmochrony_1d_a.py"

  # Your scan points (sym case): choose the ones you already explored
  seps = [20, 30, 40, 50, 60, 160]
  kappa1 = 0.2
  kappa2 = 0.2

  # Steps: small sep converges fast, but to keep uniform, we can use higher steps.
  # You can lower steps for speed if desired.
  steps = 40000
  lr = 0.02
  k_eigs = 6

  results_sym: List[RunResult] = []
  for s in seps:
    # Use more steps for sep=50 where you saw long convergence; others will stop early anyway.
    local_steps = 80000 if s == 50 else steps
    print(f"Running sep={s} ...")
    results_sym.append(run_one(TOY, sep=s, kappa1=kappa1, kappa2=kappa2, steps=local_steps, lr=lr, k_eigs=k_eigs))
    print(f"  converged={results_sym[-1].converged}, E={results_sym[-1].energy}")

  plot_sep_scan(results_sym, outfile="../figures/spectral_scan_sep.pdf", mode_indices=(0, 1, 2, 3))

  # ---- OPTIONAL: kappa2 split plot at fixed sep=50 ----
  # Uncomment to generate the degeneracy-lifting plot.
  # kappa2_list = [0.2, 0.199, 0.198, 0.195, 0.19, 0.18]
  # plot_kappa_split(TOY, sep=50, kappa1=0.2, kappa2_list=kappa2_list,
  #                  steps=80000, lr=0.02, k_eigs=6, outfile="spectral_split_kappa2.pdf",
  #                  mode_indices=(0, 1, 2, 3))
