import matplotlib.pyplot as plt
import numpy as np


def _align_trials_to_shortest(trial_returns: list[list[float]]) -> np.ndarray:
    """Convert trial return lists to a 2D array aligned by shortest trial length."""
    if not trial_returns:
        return np.empty((0, 0), dtype=float)

    min_len = min(len(curve) for curve in trial_returns)
    if min_len == 0:
        return np.empty((0, 0), dtype=float)

    aligned = [np.asarray(curve[:min_len], dtype=float) for curve in trial_returns]
    return np.vstack(aligned)


def plot_benchmark_mean_episode_returns(
    returns_by_algorithm: dict[str, list[list[float]]],
    save_path: str = "benchmark_mean_episode_returns.png",
) -> None:
    """
    Plot one mean curve per algorithm on a shared figure.

    Args:
            returns_by_algorithm: Mapping of algorithm name -> list of per-trial
                    episode return curves.
            save_path: Output path for the generated figure.

    """
    plt.figure(figsize=(10, 5))

    plotted_any = False
    for algo_name, trial_returns in returns_by_algorithm.items():
        aligned_trials = _align_trials_to_shortest(trial_returns)
        if aligned_trials.size == 0:
            continue

        plotted_any = True
        episodes = np.arange(aligned_trials.shape[1]) + 1
        mean_curve = aligned_trials.mean(axis=0)
        std_curve = aligned_trials.std(axis=0)
        line = plt.plot(episodes, mean_curve, label=f"{algo_name} (mean)")[0]
        if aligned_trials.shape[0] > 1:
            plt.fill_between(
                episodes,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.2,
                color=line.get_color(),
                linewidth=0.0,
                label=f"{algo_name} (+/-1 std)",
            )

    if not plotted_any:
        plt.close()
        return

    plt.xlabel("Episode")
    plt.ylabel("Mean episode return")
    plt.title("Learning Curves by Algorithm")
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig(save_path)
