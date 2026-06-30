from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from decent_bench.agents import Agent
from decent_bench.metrics import Metric, RuntimeMetric
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.rl_agents import RLAgentMetricsView

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.rl_agents import RLAgent


class RLMeanEpisodeReturn(Metric):
    """Mean episode return aggregated across agents."""

    description: str = "Mean Episode Return"

    def __init__(self) -> None:
        super().__init__(fmt=".2e", x_log=False, y_log=False)

    @staticmethod
    def _common_episode_count(agents: Sequence[RLAgentMetricsView]) -> int:
        return min(len(agent.episode_return_history) for agent in agents)

    def compute(
        self,
        rl_agent_view: list[RLAgentMetricsView],
        problem: "BenchmarkProblem",  # noqa: ARG002
        episode: int,
    ) -> Sequence[float]:
        """Return the mean episode return for the selected episode index."""
        n_eps = self._common_episode_count(rl_agent_view)

        if episode == -1:
            episode = n_eps - 1

        # Aggregate across agents for a single episode index.
        return [float(np.mean([agent.episode_return_history[episode] for agent in rl_agent_view]))]


class RLSmoothMeanEpisodeReturn(Metric):
    """Mean episode return aggregated across agents averaged over the last 20 episodes"""

    description: str = "Smoothen Mean Episode Return"

    def __init__(self) -> None:
        super().__init__(fmt=".2e", x_log=False, y_log=False)

    @staticmethod
    def _common_episode_count(agents: Sequence[RLAgentMetricsView]) -> int:
        return min(len(agent.episode_return_history) for agent in agents)

    def compute(
        self,
        rl_agent_view: list[RLAgentMetricsView],
        problem: "BenchmarkProblem",  # noqa: ARG002
        episode: int,
    ) -> Sequence[float]:
        """Return the mean episode return for the selected episode index averaged over the last 20 episodes."""
        n_eps = self._common_episode_count(rl_agent_view)

        if episode == -1:
            episode = n_eps - 1

        # Aggregate across agents for multiple episode indices.
        return [float(np.mean([np.mean(agent.episode_return_history[max(0, episode-49):episode + 1]) for agent in rl_agent_view]))]


class RuntimeMeanEpisodeReturn(RuntimeMetric):
    """Runtime mean episode return aggregated across all RL agents."""

    description = "Mean Episode Return"
    x_log = False
    y_log = False

    def compute(self, _problem: "BenchmarkProblem", agents: Sequence[Agent], episode: int) -> float:  # noqa: D102
        rl_agents = cast("Sequence[RLAgent]", agents)
        returns = []
        for agent in rl_agents:
            if episode < len(agent.episode_return_history):
                returns.append(float(agent.episode_return_history[episode]))
            elif agent.episode_return_history:
                returns.append(float(agent.episode_return_history[-1]))

        if len(returns) == 0:
            return float("nan")

        return float(np.mean(returns))
