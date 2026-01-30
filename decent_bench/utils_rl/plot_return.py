import numpy as np
import matplotlib.pyplot as plt

SMOOTH_WINDOW = 10


def plot_mean_episode_return(mean_episode_returns: list[float]):
    """
    Plot the mean episode return across all agents per episode.

    Args:
            mean_episode_returns: List of mean returns across agents per episode
    """
    episodes = np.arange(len(mean_episode_returns)) + 1
    returns = np.array(mean_episode_returns)

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, returns, alpha=0.4, label="mean episode return (per episode)")
    plt.xlabel("Environment steps")
    plt.ylabel("Mean episode return (across agents)")
    plt.title("IDQN learning curve (mean episode return across agents)")
    plt.legend()
    plt.grid(True)
    plt.savefig("idqn_mean_episode_returns.png")
