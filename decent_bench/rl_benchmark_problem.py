from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3


@dataclass(eq=False)
class RLBenchmarkProblem:
    """
    RL benchmark problem describing environment setup.

    Args:
        env_factory: callable PettingZoo environment
        n_agents: number of agents in the environment
        The problem intentionally does not own training-loop settings such as
        episode length or number of episodes.

    """

    env_factory: Callable[..., Any]
    n_agents: int

def create_simple_spread_problem(
    n_agents: int = 1,
) -> RLBenchmarkProblem:
    """
    Create out-of-the-box simple-spread problems using PettingZoo.

    Args:
        n_agents: number of agents in the environment

    """

    def env_factory(**env_kwargs: Any) -> Any:  # noqa: ANN401
        return simple_spread_v3.parallel_env(
            N=n_agents,
            render_mode="human",
            **env_kwargs,
        )

    return RLBenchmarkProblem(
        env_factory=env_factory,
        n_agents=n_agents,
    )


def create_simple_adversary_problem(
    n_good_agents: int = 2,
) -> RLBenchmarkProblem:
    """
    Create an out-of-the-box simple-adversary problem using PettingZoo.

    Args:
        n_good_agents: number of cooperative (non-adversary) agents.

    """
    # simple_adversary_v3 is used with one adversary by design.
    total_agents = n_good_agents + 1

    def env_factory(**env_kwargs: Any) -> Any:  # noqa: ANN401
        return simple_adversary_v3.parallel_env(
            N=n_good_agents,
            render_mode="human",
            **env_kwargs,
        )

    return RLBenchmarkProblem(
        env_factory=env_factory,
        n_agents=total_agents,
    )
