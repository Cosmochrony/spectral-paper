# fig_deltaa_vs_neg_dlogdet_by_radius_inset.py
# Publication-ready PDF with an inset zoom near Δa ≈ 0.
#
# Usage:
#   python fig_deltaa_vs_neg_dlogdet_by_radius_inset.py scan_deltaa_logdet.csv
#
# Output:
#   fig_deltaa_vs_neg_dlogdet_by_radius_inset.pdf

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"radius", "factor", "delta_a", "delta_logdet_local"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")
    df = df.copy()
    df["radius"] = df["radius"].astype(int)
    df["factor"] = df["factor"].astype(float)
    return df


def plot_curves(ax: plt.Axes, df: pd.DataFrame) -> None:
    markers = {1: "o", 2: "s", 3: "^"}
    linestyles = {1: "-", 2: "--", 3: "-."}

    for rad in sorted(df["radius"].unique()):
        sub = df.loc[df["radius"] == rad].sort_values(by="factor", ascending=False)
        x = (-sub["delta_logdet_local"]).to_numpy(dtype=float)
        y = sub["delta_a"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker=markers.get(int(rad), "o"),
            linestyle=linestyles.get(int(rad), "-"),
            linewidth=1.2,
            markersize=5,
            label=f"radius={rad}",
        )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python fig_deltaa_vs_neg_dlogdet_by_radius_inset.py scan_deltaa_logdet.csv")

    df = load_csv(sys.argv[1])

    fig = plt.figure(figsize=(6.4, 4.4))
    ax = fig.add_subplot(111)

    ax.axhline(0.0, linewidth=1.0)
    plot_curves(ax, df)

    ax.set_xlabel(r"$-\Delta \log\det_{\mathrm{local}}(A)$")
    ax.set_ylabel(r"$\Delta a$ from $a/r$ fit")

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(yfmt)

    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.20)
    ax.legend(frameon=False, loc="best")

    # ---- inset zoom (auto window based on radii 1 and 2) ----
    df_small = df[df["radius"].isin([1, 2])]
    x_small = (-df_small["delta_logdet_local"]).to_numpy(dtype=float)
    y_small = df_small["delta_a"].to_numpy(dtype=float)

    # Expand ranges a bit for readability
    x_pad = 0.08 * (x_small.max() - x_small.min() + 1e-12)
    y_pad = 0.25 * (y_small.max() - y_small.min() + 1e-18)

    x1 = max(0.0, float(x_small.min() - x_pad))
    x2 = float(x_small.max() + x_pad)

    y1 = float(y_small.min() - y_pad)
    y2 = float(y_small.max() + y_pad)

    axins = inset_axes(ax, width="45%", height="45%", loc="upper left", borderpad=1.2)
    axins.axhline(0.0, linewidth=1.0)
    plot_curves(axins, df)

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.xaxis.set_minor_locator(AutoMinorLocator(2))
    axins.yaxis.set_minor_locator(AutoMinorLocator(2))
    axins.grid(True, which="major", linewidth=0.5, alpha=0.30)
    axins.grid(True, which="minor", linewidth=0.4, alpha=0.18)

    # Draw connectors
    mark_inset(ax, axins, loc1=2, loc2=4, linewidth=0.8)

    fig.tight_layout()
    fig.savefig("fig_deltaa_vs_neg_dlogdet_by_radius_inset.pdf")
    plt.close(fig)

    print("Wrote: fig_deltaa_vs_neg_dlogdet_by_radius_inset.pdf")


if __name__ == "__main__":
    main()