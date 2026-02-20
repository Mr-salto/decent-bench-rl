from typing import List, Callable, Any
from dataclasses import dataclass


from pettingzoo.mpe import simple_spread_v3

from decent_bench.rl_agents import RLAgent


@dataclass(eq=False)
class RLBenchmarkProblem:
    """
    RL Benchmark problem to run algorithms on, defining settings such as environment type, episode length or number of agents.

    Args:
        env_factory: callable PettingZoo environment
        agents: number of agents
        episode_length: number of steps taken in each episode
        n_episodes: number of episodes
    """
    env_factory: Callable[..., Any]
    n_agents: int
    episode_length: int
    n_episodes: int



def create_simple_spread_problem(
    n_agents: int = 1,
    episode_length: int = 30,
    n_episodes: int = 100
) -> RLBenchmarkProblem:
    """
    Create out-of-the-box simple-spread problems using PettingZoo.

    Args:
        agents: list of agents
        episode_length: number of steps taken in each episode
        n_episodes: number of episodes
    """
    env_factory = lambda: simple_spread_v3.parallel_env(N=n_agents, render_mode="human", max_cycles=episode_length - 1)
    return RLBenchmarkProblem(
        env_factory = env_factory,
        n_agents = n_agents,
        episode_length = episode_length,
        n_episodes = n_episodes
    )
