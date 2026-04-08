from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from decent_bench.agents import AgentMetricsView
from decent_bench.metrics import Metric, X, Y
from decent_bench.metrics._metric import Statistic
from decent_bench.rl_agents import RLAgentMetricsView

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem


class RLMeanEpisodeReturn(Metric):
    """Mean episode return aggregated across agents."""

    table_description: str = "Mean Episode Return"
    plot_description: str = "Mean Episode Return"

    def __init__(self, statistics: Sequence[Statistic] = (np.average,)) -> None:
        super().__init__(statistics=statistics, fmt=".2e", x_log=False, y_log=False)

    @staticmethod
    def _common_episode_count(agents: Sequence[RLAgentMetricsView]) -> int:
        return min(len(agent.episode_return_history) for agent in agents)

    def get_data_from_trial(
        self,
        agents: Sequence[AgentMetricsView],
        problem: "BenchmarkProblem",  # noqa: ARG002
        iteration: int,
    ) -> Sequence[float]:
        """Return per-trial mean return for the selected episode index."""
        rl_agents = cast("Sequence[RLAgentMetricsView]", agents)
        n_eps = self._common_episode_count(rl_agents)

        if iteration == -1:
            iteration = n_eps - 1

        # Aggregate across agents for a single episode index.
        return [float(np.mean([agent.episode_return_history[iteration] for agent in rl_agents]))]

    def get_plot_data(
        self,
        agents: Sequence[AgentMetricsView],
        problem: "BenchmarkProblem",  # noqa: ARG002
    ) -> Sequence[tuple[X, Y]]:
        """Return episode-indexed mean returns for plotting."""
        rl_agents = cast("Sequence[RLAgentMetricsView]", agents)
        n_eps = self._common_episode_count(rl_agents)

        # X-axis uses 1-based episode number for readability.
        return [
            (float(ep + 1), float(np.mean([agent.episode_return_history[ep] for agent in rl_agents])))
            for ep in range(n_eps)
        ]

    def get_table_data(
        self,
        agents: Sequence[AgentMetricsView],
        problem: "BenchmarkProblem",  # noqa: ARG002
    ) -> Sequence[float]:
        """Return the final aggregated mean return for table display."""
        rl_agents = cast("Sequence[RLAgentMetricsView]", agents)
        n_eps = self._common_episode_count(rl_agents)

        episode_means = [
            float(np.mean([agent.episode_return_history[ep] for agent in rl_agents])) for ep in range(n_eps)
        ]
        return [episode_means[-1]]
