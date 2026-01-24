import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"

def sample_on_S3(n_samples, rng):
    """
    Uniformly samples points on S^3
    by normalizing a 4D Gaussian distribution.
    """
    vecs = rng.normal(size=(n_samples, 4))
    vecs /= np.linalg.norm(vecs, axis=1)[:, None]
    return vecs


def calculate_ratio_and_moments(n_samples, rng):
    """
    Computes:
      - ⟨cos²⟩
      - ⟨sin²⟩
      - R = 8 ⟨cos²⟩ / ⟨sin²⟩
    """
    vecs = sample_on_S3(n_samples, rng)

    # Fiber axis (arbitrary choice, due to isotropy of S^3)
    fiber_axis = np.array([1.0, 0.0, 0.0, 0.0])

    cos_sq = np.dot(vecs, fiber_axis) ** 2
    sin_sq = 1.0 - cos_sq

    mean_cos = cos_sq.mean()
    mean_sin = sin_sq.mean()

    ratio = 8.0 * mean_cos / mean_sin

    return ratio, mean_cos, mean_sin


def run_convergence_study(repeats=30, seed=0):
    rng = np.random.default_rng(seed)

    resolutions = np.logspace(2, 6, 15, dtype=int)

    ratios_mean = []
    ci_low = []
    ci_high = []

    print("Monte Carlo convergence toward 8/3 on S³\n")

    for n in resolutions:
        ratios = []
        cos_vals = []
        sin_vals = []

        for _ in range(repeats):
            r, c, s = calculate_ratio_and_moments(n, rng)
            ratios.append(r)
            cos_vals.append(c)
            sin_vals.append(s)

        ratios = np.array(ratios)

        mean_r = ratios.mean()
        std_r = ratios.std(ddof=1)
        ci = 1.96 * std_r / np.sqrt(repeats)

        ratios_mean.append(mean_r)
        ci_low.append(mean_r - ci)
        ci_high.append(mean_r + ci)

        print(
            f"N={n:7d} | "
            f"mean={mean_r:.6f} | "
            f"sd={std_r:.6f} | "
            f"95%CI=±{ci:.6f}"
        )

    # ------------------
    # Visualization (PDF)
    # ------------------

    plt.figure(figsize=(10, 6))

    plt.axhline(
      y=8 / 3,
      linestyle="--",
      color="black",
      label="Theoretical target (8/3)"
    )

    plt.plot(
      resolutions,
      ratios_mean,
      "o-",
      label="Monte Carlo mean"
    )

    plt.fill_between(
      resolutions,
      ci_low,
      ci_high,
      alpha=0.25,
      label="95% CI (over repeats)"
    )

    plt.xscale("log")
    plt.xlabel("Resolution (N)")
    plt.ylabel("Ratio R")
    plt.title("Monte Carlo convergence toward 8/3 on S³")
    plt.legend()
    plt.grid(True, which="both", alpha=0.5)
    plt.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)

    plt.savefig(
      FIG_DIR / "convergence_8_3_S3.pdf",
      format="pdf",
      bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":
    run_convergence_study()
