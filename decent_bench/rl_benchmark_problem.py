from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3


@dataclass(eq=False)
class RLBenchmarkProblem:
    """
    RL benchmark problem describing environment setup and episode settings.

    Args:
        env_factory: callable PettingZoo environment
        n_agents: number of agents in the environment
        episode_length: number of steps taken in each episode

    """

    env_factory: Callable[..., Any]
    n_agents: int
    episode_length: int


def create_simple_spread_problem(
    n_agents: int = 1,
    episode_length: int = 50,
) -> RLBenchmarkProblem:
    """
    Create out-of-the-box simple-spread problems using PettingZoo.

    Args:
        n_agents: number of agents in the environment
        episode_length: number of steps taken in each episode

    """

    def env_factory() -> Any:  # noqa: ANN401
        return simple_spread_v3.parallel_env(
            N=n_agents,
            render_mode="human",
            max_cycles=episode_length - 1,
        )

    return RLBenchmarkProblem(
        env_factory=env_factory,
        n_agents=n_agents,
        episode_length=episode_length,
    )


def create_simple_adversary_problem(
    n_good_agents: int = 2,
    episode_length: int = 50,
) -> RLBenchmarkProblem:
    """
    Create an out-of-the-box simple-adversary problem using PettingZoo.

    Args:
        n_good_agents: number of cooperative (non-adversary) agents.
        episode_length: number of steps taken in each episode.

    """
    # simple_adversary_v3 is used with one adversary by design.
    total_agents = n_good_agents + 1

    def env_factory() -> Any:  # noqa: ANN401
        return simple_adversary_v3.parallel_env(
            N=n_good_agents,
            render_mode="human",
            max_cycles=episode_length - 1,
        )

    return RLBenchmarkProblem(
        env_factory=env_factory,
        n_agents=total_agents,
        episode_length=episode_length,
    )
