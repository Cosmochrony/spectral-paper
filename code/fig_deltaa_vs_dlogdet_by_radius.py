# fig_deltaa_vs_dlogdet_by_radius.py
# Publication-ready PDF: 3 curves (one per radius): Δa vs -Δlogdet_local(A)
#
# Usage:
#   python fig_deltaa_vs_dlogdet_by_radius.py scan_deltaa_logdet.csv
#
# Output:
#   fig_deltaa_vs_neg_dlogdet_by_radius.pdf

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter


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

    annotate_factor = False  # set True if you want to print factors near points

    plt.figure(figsize=(6.2, 4.2))
    ax = plt.gca()

    ax.axhline(0.0, linewidth=1.0)

    markers = {1: "o", 2: "s", 3: "^"}
    linestyles = {1: "-", 2: "--", 3: "-."}

    for rad in sorted(df["radius"].unique()):
        sub = (
          df.loc[df["radius"] == rad]
          .sort_values(by="factor", ascending=False)
        )

        x = -sub["delta_logdet_local"].to_numpy()
        y = sub["delta_a"].to_numpy()

        plt.plot(x, y, marker="o", linestyle="-", label=f"radius={rad}")

        if annotate_factor:
            for xi, yi, fi in zip(x, y, f):
                ax.annotate(f"{fi:g}", (xi, yi), textcoords="offset points", xytext=(4, 3), fontsize=8)

    ax.set_xlabel(r"$-\Delta \log\det_{\mathrm{local}}(A)$")
    ax.set_ylabel(r"$\Delta a$ from $a/r$ fit")

    # Ticks and grid (paper-friendly)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.20)

    # Scientific notation on y if needed
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(yfmt)

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("fig_deltaa_vs_neg_dlogdet_by_radius.pdf")
    plt.close()

    print("Wrote: fig_deltaa_vs_neg_dlogdet_by_radius.pdf")


if __name__ == "__main__":
    main()