import numpy as np
import matplotlib.pyplot as plt


def sample_on_S3_biased(n_samples, alpha, rng):
    """
    Biased sampling on S^3.
    Probability density ∝ exp(alpha * cos²θ).
    """
    accepted = []
    fiber_axis = np.array([1.0, 0.0, 0.0, 0.0])

    # Upper bound of the weight (cos² <= 1)
    w_max = np.exp(alpha)

    while len(accepted) < n_samples:
        v = rng.normal(size=4)
        v /= np.linalg.norm(v)

        cos_sq = np.dot(v, fiber_axis) ** 2
        w = np.exp(alpha * cos_sq)

        if rng.random() < w / w_max:
            accepted.append(v)

    return np.array(accepted)


def calculate_ratio_biased(n_samples, alpha, rng):
    vecs = sample_on_S3_biased(n_samples, alpha, rng)

    cos_sq = vecs[:, 0] ** 2
    sin_sq = 1.0 - cos_sq

    return 8.0 * cos_sq.mean() / sin_sq.mean()


def run_bias_study():
    rng = np.random.default_rng(0)

    alphas = np.linspace(-2.0, 2.0, 17)
    n_samples = 100_000
    repeats = 10

    ratios = []
    errs = []

    print("Relaxation bias study\n")
    for alpha in alphas:
        vals = []
        for _ in range(repeats):
            vals.append(calculate_ratio_biased(n_samples, alpha, rng))

        vals = np.array(vals)
        mean = vals.mean()
        err = 1.96 * vals.std(ddof=1) / np.sqrt(repeats)

        ratios.append(mean)
        errs.append(err)

        print(
            f"alpha={alpha:+.2f} | "
            f"R={mean:.6f} ± {err:.6f}"
        )

    # ------------------
    # Visualization
    # ------------------

    plt.figure(figsize=(9, 6))

    plt.axhline(
        y=8 / 3,
        linestyle="--",
        color="black",
        label="Isotropy (8/3)"
    )

    plt.errorbar(
        alphas,
        ratios,
        yerr=errs,
        fmt="o-",
        capsize=4,
        label="Biased relaxation"
    )

    plt.xlabel("Relaxation bias α")
    plt.ylabel("Ratio R")
    plt.title("Deviation of the cosmochrony ratio under relaxation bias")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_bias_study()
