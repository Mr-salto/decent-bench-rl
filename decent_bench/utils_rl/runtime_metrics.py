from __future__ import annotations

from copy import deepcopy
from multiprocessing import get_context
from typing import TYPE_CHECKING, Any, cast

from decent_bench.metrics import RuntimeMetricPlotter
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    import queue
    from collections.abc import Sequence
    from multiprocessing.context import SpawnContext

    from decent_bench.agents import Agent
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.metrics import RuntimeMetric
    from decent_bench.rl_benchmark import RLBenchmarkProblem


def start_runtime_metric_plotter(
    runtime_metrics: list[RuntimeMetric] | None,
    mp_context: SpawnContext | None = None,
) -> tuple[RuntimeMetricPlotter | None, queue.Queue[Any] | None]:
    """Create and start a centralized runtime metric plotter for RL benchmarks."""
    if not runtime_metrics:
        return None, None

    ctx = mp_context if mp_context is not None else get_context()
    runtime_queue = ctx.Manager().Queue()

    runtime_plotter = RuntimeMetricPlotter(runtime_queue, ctx)
    runtime_plotter.start()
    return runtime_plotter, runtime_queue


def prepare_trial_runtime_metrics(
    runtime_metrics: list[RuntimeMetric] | None,
    algorithm_name: str,
    trial: int,
    runtime_queue: queue.Queue[Any] | None = None,
) -> list[RuntimeMetric]:
    """Initialize per-trial runtime metrics with isolated metric instances."""
    if not runtime_metrics:
        return []

    if runtime_queue is None:
        LOGGER.warning("Runtime metrics provided but no runtime queue available, metrics will not be plotted")
        return []

    trial_runtime_metrics = [deepcopy(metric) for metric in runtime_metrics]
    for metric in trial_runtime_metrics:
        try:
            metric.initialize_plot(algorithm_name, trial, runtime_queue)
        except Exception as e:
            LOGGER.warning(f"Failed to initialize runtime metric {metric.description}: {e}")

    return trial_runtime_metrics


def update_trial_runtime_metrics(
    runtime_metrics: Sequence[RuntimeMetric],
    problem: RLBenchmarkProblem,
    agents: Sequence[Agent],
    episode: int,
    total_episodes: int,
) -> None:
    """Update RL runtime metrics at episode boundaries."""
    for metric in runtime_metrics:
        if metric.should_update(episode) or episode + 1 == total_episodes:
            try:
                metric.update_plot(cast("BenchmarkProblem", problem), agents, episode)
            except Exception as e:
                LOGGER.warning(f"Failed to update runtime metric {metric.description} at episode {episode}: {e}")
