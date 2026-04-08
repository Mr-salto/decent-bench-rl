import logging
from json import JSONDecodeError
from typing import TYPE_CHECKING, cast

import numpy as np

from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark import BenchmarkProblem
from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics import Metric, compute_plots, compute_tables
from decent_bench.networks import Network
from decent_bench.rl_agents import RLAgentMetricsView
from decent_bench.rl_benchmark._rl_benchmark_result import RLBenchmarkResult
from decent_bench.rl_metrics import RLMeanEpisodeReturn
from decent_bench.utils import logger

if TYPE_CHECKING:
    from decent_bench.utils_rl.rl_checkpoint_manager import RLCheckpointManager


DEFAULT_RL_TABLE_METRICS: list[Metric] = [RLMeanEpisodeReturn(statistics=(np.average,))]
DEFAULT_RL_PLOT_METRICS: list[Metric] = [RLMeanEpisodeReturn(statistics=(np.average,))]


def compute_metrics(
    benchmark_result: RLBenchmarkResult | None = None,
    checkpoint_manager: "RLCheckpointManager | None" = None,
    *,
    table_metrics: list[Metric] = DEFAULT_RL_TABLE_METRICS,
    plot_metrics: list[Metric] | list[list[Metric]] = DEFAULT_RL_PLOT_METRICS,
    confidence_level: float = 0.95,
    log_level: int = logging.INFO,
) -> MetricResult:
    """
    Compute RL metrics from a ``RLBenchmarkResult``.

    This function reuses the shared metric computation pipeline used by the non-RL benchmark.
    RL metric implementations are expected to override ``get_plot_data``/``get_table_data``
    when they do not rely on optimization-specific ``x_history``.

    Args:
        benchmark_result: result of an RL benchmark execution.
        checkpoint_manager: if provided, will be used to save results of metrics computation and load benchmark
            result.
        table_metrics: metrics to tabulate as confidence intervals.
        plot_metrics: metrics to compute for plots.
        confidence_level: confidence level for table metrics.
        log_level: minimum log level.

    Raises:
        ValueError: If neither ``benchmark_result`` nor ``checkpoint_manager`` is provided, or
            if the checkpoint manager does not contain a valid benchmark result to load.

    Returns:
        MetricResult containing computed table and plot metric data.

    """
    logger.start_logger(log_level=log_level)

    if benchmark_result is None:
        if checkpoint_manager is None:
            raise ValueError(
                "If ``benchmark_result`` is not provided, ``checkpoint_manager`` must be provided "
                "to load the benchmark result from."
            )
        try:
            benchmark_result = checkpoint_manager.load_benchmark_result()
        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"Invalid checkpoint directory: missing or corrupted metadata - {e}") from e
        except JSONDecodeError as e:
            raise ValueError(f"Invalid checkpoint directory: metadata is not valid JSON - {e}") from e

        if len(benchmark_result.result) == 0:
            raise ValueError("No benchmark result found in checkpoint manager to compute metrics")

    resulting_agent_states = {
        alg: [[RLAgentMetricsView.from_agent(agent) for agent in trial_agents] for trial_agents, _ in trials]
        for alg, trials in benchmark_result.result.items()
    }
    shared_resulting_agent_states = cast(
        "dict[Algorithm[Network], list[list[AgentMetricsView]]]",
        resulting_agent_states,
    )
    shared_problem = cast("BenchmarkProblem", benchmark_result.problem)

    # The shared metric engine works as long as metrics know how to interpret the view objects they receive.
    table_results = compute_tables(
        shared_resulting_agent_states,
        shared_problem,
        table_metrics,
        confidence_level,
    )
    plot_results = compute_plots(
        shared_resulting_agent_states,
        shared_problem,
        plot_metrics,
    )

    result = MetricResult(
        agent_metrics=shared_resulting_agent_states,
        table_metrics=table_metrics,
        plot_metrics=plot_metrics,
        table_results=table_results,
        plot_results=plot_results,
    )

    if checkpoint_manager is not None:
        checkpoint_manager.save_metrics_result(result)

        if any(isinstance(m, list) for m in plot_metrics):
            flat_plot_metrics = [metric for group in plot_metrics for metric in group]  # type: ignore[union-attr]
        else:
            flat_plot_metrics = plot_metrics  # type: ignore[assignment]

        metadata = {
            "rl_table_metrics": [metric.table_description for metric in table_metrics],
            "rl_plot_metrics": [metric.plot_description for metric in flat_plot_metrics],
        }
        checkpoint_manager.append_metadata(metadata)

    return result
