# fig_deltaa_vs_dlogdet_by_radius.py
# Generate a publication-ready PDF with 3 curves (one per radius):
#   Δa vs Δlogdet_local(A)
#
# Usage:
#   python fig_deltaa_vs_dlogdet_by_radius.py scan_deltaa_logdet.csv
#
# Output:
#   fig_deltaa_vs_dlogdet_by_radius.pdf

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"radius", "factor", "delta_a", "delta_logdet_local"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")
    return df


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python fig_deltaa_vs_dlogdet_by_radius.py scan_deltaa_logdet.csv")

    df = load_csv(sys.argv[1]).copy()
    df["radius"] = df["radius"].astype(int)
    df["factor"] = df["factor"].astype(float)

    plt.figure(figsize=(6.2, 4.2))
    plt.axhline(0.0, linewidth=1.0)

    markers = {1: "o", 2: "s", 3: "^"}
    linestyles = {1: "-", 2: "--", 3: "-."}

    for rad in sorted(df["radius"].unique()):
        sub = df[df["radius"] == rad].sort_values("factor", ascending=False)

        x = sub["delta_logdet_local"].to_numpy(dtype=float)
        y = sub["delta_a"].to_numpy(dtype=float)

        plt.plot(
            x,
            y,
            marker=markers.get(int(rad), "o"),
            linestyle=linestyles.get(int(rad), "-"),
            linewidth=1.2,
            markersize=5,
            label=f"radius={rad}",
        )

    plt.xlabel(r"$\Delta \log\det_{\mathrm{local}}(A)$")
    plt.ylabel(r"$\Delta a$ from $a/r$ fit")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("fig_deltaa_vs_dlogdet_by_radius.pdf")
    plt.close()

    print("Wrote: fig_deltaa_vs_dlogdet_by_radius.pdf")


if __name__ == "__main__":
    main()