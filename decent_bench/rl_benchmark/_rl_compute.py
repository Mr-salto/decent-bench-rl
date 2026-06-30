import logging
from collections.abc import Callable, Mapping, Sequence
from json import JSONDecodeError
from typing import TYPE_CHECKING, Mapping, cast

from rich.status import Status
import numpy as np
import pandas as pd


from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.algorithms import Algorithm
from decent_bench.metrics import Metric, utils
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.networks import Network
from decent_bench.rl_agents import RLAgentMetricsView
from decent_bench.rl_algorithms import RLAlgorithm
from decent_bench.rl_benchmark._rl_benchmark_result import RLBenchmarkResult
from decent_bench.rl_metrics import RLMeanEpisodeReturn, RLSmoothMeanEpisodeReturn
from decent_bench.utils.logger import LOGGER, start_logger

if TYPE_CHECKING:
    from decent_bench.utils_rl.rl_checkpoint_manager import RLCheckpointManager
    from decent_bench.rl_benchmark._rl_benchmark_problem import RLBenchmarkProblem



DEFAULT_RL_TABLE_METRICS: list[Metric] = [RLMeanEpisodeReturn(), RLSmoothMeanEpisodeReturn()]
DEFAULT_RL_PLOT_METRICS: list[Metric] = [RLMeanEpisodeReturn(), RLSmoothMeanEpisodeReturn()]


def compute_metrics(
    benchmark_result: RLBenchmarkResult | None = None,
    checkpoint_manager: "RLCheckpointManager | None" = None,
    *,
    table_metrics: list[Metric] = DEFAULT_RL_TABLE_METRICS,
    plot_metrics: list[Metric] | list[list[Metric]] = DEFAULT_RL_PLOT_METRICS,
    statistics_across_agents: list[str] | None = None,
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
        statistics_across_agents: list of statistics to compute across agents for metrics that return one value per agent.
        log_level: minimum log level.

    Raises:
        ValueError: If neither ``benchmark_result`` nor ``checkpoint_manager`` is provided, or
            if the checkpoint manager does not contain a valid benchmark result to load.

    Returns:
        MetricResult containing the computed metrics.

    """
    start_logger(log_level=log_level)
    LOGGER.info("Starting metrics computation")
    
    # 1) user input validation
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

    # 2) compute table and plot metrics
    rl_agent_views = {
        alg: [[RLAgentMetricsView.from_agent(agent) for agent in trial_agents] for trial_agents, _ in trials]
        for alg, trials in benchmark_result.result.items()
    }

    episodes = _all_episodes(rl_agent_views)

    # compute metrics
    raw_plot_results = _compute_plot_metrics(rl_agent_views, benchmark_result.problem, plot_metrics, episodes)
    raw_table_results = _compute_table_metrics(
        rl_agent_views, benchmark_result.problem, table_metrics, episodes, raw_plot_results
    )

    # aggregate metrics
    aggregated_plot_metrics = _aggregate_plot_metrics(raw_plot_results)
    aggregated_table_metrics = _aggregate_table_metrics(raw_table_results, statistics_across_agents)

    network_views = cast("dict[Algorithm[Network], Sequence[NetworkMetricsView]]", rl_agent_views)

    result = MetricResult(
        network_views=network_views,
        raw_table_results=raw_table_results,
        raw_plot_results=raw_plot_results,
        table_results=aggregated_table_metrics,
        plot_results=aggregated_plot_metrics,
    )

    if checkpoint_manager is not None:
        with Status("Saving computed metrics..."):
            metadata = {
                "table_metrics": [metric.description for metric in table_metrics],
                "plot_metrics": [metric.description for metric in plot_metrics],
            }
            checkpoint_manager.save_metrics_result(result)
            checkpoint_manager.append_metadata(metadata)

    return result


def _compute_plot_metrics(
    rl_agent_views: dict[RLAlgorithm, list[list[RLAgentMetricsView]]],
    problem: "RLBenchmarkProblem",
    metrics: list[Metric],
    episodes: list[int],
) -> Mapping[Metric, pd.DataFrame]:
    """
    Compute metrics at each episode and return one DataFrame per metric.

    The DataFrame has columns (algorithm, trial, agent, iteration, value).
    """
    frames_by_metric: dict[Metric, pd.DataFrame] = {}
    if not episodes:
        return frames_by_metric
    total_tasks = len(metrics) * len(episodes)

    with utils.MetricProgressBar() as progress:
        plot_task = progress.add_task("Computing plot metrics", total=total_tasks, status="")

        for metric in metrics:
            progress.update(plot_task, status=f"Task: {metric.description}")

            frames_by_episodes = []
            for episode in episodes:
                frames_by_episodes.append(_compute_metrics_at_ep(rl_agent_views, problem, metric, episode))
                progress.advance(plot_task)

            frames_by_metric[metric] = pd.concat(frames_by_episodes, ignore_index=True)

        progress.update(plot_task, status="Plot computation complete")

    return frames_by_metric

def _aggregate_plot_metrics(
    plot_results: Mapping[Metric, pd.DataFrame],
) -> pd.DataFrame | None:
    """
    Aggregate plot metrics by mean across agents and by mean, min, and max across trials.

    The DataFrame that is returned has columns (metric, algorithm, iteration, mean, min, max), with mean, min, max being
    float32.
    """
    frames_by_metric: list[pd.DataFrame] = []

    for metric, frame in plot_results.items():
        # 1) take mean of values across agents
        new_frame = frame.groupby(["algorithm", "trial", "iteration"])["value"].mean().reset_index()

        # 2) compute mean, min, max across trials
        new_frame = (
            new_frame.groupby(["algorithm", "iteration"])["value"].agg(mean="mean", min="min", max="max").reset_index()
        )

        # 3) add metric column
        new_frame = new_frame.assign(metric=metric.description)
        new_frame = new_frame[["metric", "algorithm", "iteration", "mean", "min", "max"]]  # reorder columns
        new_frame[["mean", "min", "max"]] = new_frame[["mean", "min", "max"]].astype("float32")

        frames_by_metric.append(new_frame)

    # 4) return concatenated frames
    if not frames_by_metric:
        return None
    return pd.concat(frames_by_metric, ignore_index=True)


STATISTICS: dict[str, Callable[[Sequence[float]], float]] = {
    "mean": np.mean,
    "std": np.std,
    "max": np.max,
    "min": np.min,
    "median": np.median,
}
STATISTICS_ALIASES = {
    "average": "mean",
    "avg": "mean",
    "maximum": "max",
    "minimum": "min",
    "mdn": "median",
}
DEFAULT_STATISTICS = {"mean": STATISTICS["mean"], "std": STATISTICS["std"]}


def _resolve_statistics(statistics: list[str] | None) -> dict[str, Callable[[Sequence[float]], float]]:
    """Resolve statistics, defaulting to mean and std if they cannot be resolved."""
    # default to mean and std
    if statistics is None:
        return DEFAULT_STATISTICS

    resolved_stats: list[str] = []
    for stat in statistics:
        resolved_stat = STATISTICS_ALIASES.get(stat, stat)
        if resolved_stat not in STATISTICS:
            LOGGER.warning(f"Skipping {stat} because it is not a valid statistic (or alias)")
            continue
        resolved_stats.append(resolved_stat)

    if not resolved_stats:
        LOGGER.warning(
            f"No valid statistic was passed, defaulting to mean and std; "
            f"valid stats: {', '.join(STATISTICS.keys())}, passed: {', '.join(statistics)}"
        )
        return DEFAULT_STATISTICS

    return {name: STATISTICS[name] for name in resolved_stats}


def _compute_table_metrics(
    rl_agent_views: dict[RLAlgorithm, list[list[RLAgentMetricsView]]],
    problem: "RLBenchmarkProblem",
    metrics: list[Metric],
    episodes: list[int],
    plot_metrics_results: Mapping[Metric, pd.DataFrame] | None = None,
) -> Mapping[Metric, pd.DataFrame]:
    """
    Compute metrics at the final episode and return one DataFrame per metric.

    If ``plot_metrics_results`` is not None and contains a metric from *metrics*, the DataFrame is extracted from
    ``plot_metrics_results`` instead of recomputing it.
    The DataFrame has columns (algorithm, trial, agent, value), since the iteration column is dropped.
    """
    if not plot_metrics_results:
        plot_metrics_results = {}
    frames_by_metric: dict[Metric, pd.DataFrame] = {}
    already_computed: set[Metric] = set() if not plot_metrics_results else set(metrics) & set(plot_metrics_results)
    if not episodes:
        return frames_by_metric
    final_episode = max(episodes)

    with utils.MetricProgressBar() as progress:
        table_task = progress.add_task("Computing table metrics", total=len(metrics), status="")

        for metric in metrics:
            progress.update(table_task, status=f"Task: {metric.description}")

            if metric in already_computed:
                plot_frame = plot_metrics_results[metric]
                frames_by_metric[metric] = plot_frame.loc[plot_frame["iteration"] == final_episode]
            else:
                frames_by_metric[metric] = _compute_metrics_at_ep(rl_agent_views, problem, metric, final_episode)
            progress.advance(table_task)

            frames_by_metric[metric] = frames_by_metric[metric].drop("iteration", axis="columns")

        progress.update(table_task, status="Table computation complete")

    return frames_by_metric


def _aggregate_table_metrics(
    table_results: Mapping[Metric, pd.DataFrame],
    statistics: list[str] | None = None,
) -> pd.DataFrame | None:
    """
    Aggregate table metrics by statistics across agents and by mean, std across trials.

    The DataFrame that is returned has columns (metric, algorithm, statistic, mean, std). Numerical values are cast
    to float32.
    """
    resolved_statistics = _resolve_statistics(statistics)
    frames_by_metric: list[pd.DataFrame] = []

    for metric, frame in table_results.items():
        # if there is a single value (agent is always 0), drop agent column and add a dummy statistic column
        if len(frame["agent"].unique()) == 1:
            new_frame = frame.loc[:, ["algorithm", "trial", "value"]].copy()
            new_frame["statistic"] = ""
            new_frame = new_frame[["algorithm", "trial", "statistic", "value"]]  # reorder columns

        # if there are per-agent values, compute statistics across them
        # gives DataFrame with columns (algorithm, trial, statistic, value)
        else:
            # compute statistics
            agg_spec = {name: ("value", func) for name, func in resolved_statistics.items()}
            new_frame = frame.groupby(["algorithm", "trial"]).agg(**agg_spec).reset_index()
            # turn the statistics into a new column
            new_frame = new_frame.melt(
                id_vars=["algorithm", "trial"],
                value_vars=list(resolved_statistics.keys()),
                var_name="statistic",
                value_name="value",
            )

        # 3) compute mean and std across trials, gives DataFrame with (algorithm, statistic, mean, std)
        new_frame = (
            new_frame.groupby(["algorithm", "statistic"])["value"]
            .agg(mean="mean", std=lambda x: x.std(ddof=0))
            .reset_index()
        )

        # 4) add metric column
        new_frame = new_frame.assign(metric=metric.description)
        new_frame = new_frame[["metric", "algorithm", "statistic", "mean", "std"]]  # reorder columns
        new_frame[["mean", "std"]] = new_frame[["mean", "std"]].astype("float32")

        frames_by_metric.append(new_frame)

    # 5) return concatenated frames
    if not frames_by_metric:
        return None
    return pd.concat(frames_by_metric, ignore_index=True)


MAX_ABS_METRIC_VALUE = 1e30

def _compute_metrics_at_ep(
    rl_agent_views: dict[RLAlgorithm, list[list[RLAgentMetricsView]]],
    problem: "RLBenchmarkProblem",
    metric: Metric,
    episode: int,
) -> pd.DataFrame:
    """
    Compute the metric at the given episode for each algorithm and each trial.

    The function returns a DataFrame with columns (algorithm, trial, agent, iteration, value) which are of types
    (str, uint16, uint16, uint32, float32); algorithm is categorical. The elements in the
    value column are set to +/-inf if value>1e30/<-1e30, and NaN is set to inf.
    """
    # 1) compute metrics across algorithms and trials
    rows: list[dict[str, object]] = []

    for algorithm, rl_agent_views_by_trials in rl_agent_views.items():
        for trial_idx, rl_agent_view in enumerate(rl_agent_views_by_trials):
            data = metric.compute(rl_agent_view, problem, episode)

            for agent_idx, value in enumerate(data):
                rows.append(
                    {
                        "algorithm": algorithm.name,
                        "trial": trial_idx,
                        "agent": agent_idx,
                        "iteration": episode,
                        "value": value,
                    }
                )

    # 2) create dataframe, remove extreme values (or NaN), and cast columns to appropriate dtypes
    frame = pd.DataFrame.from_records(rows, columns=["algorithm", "trial", "agent", "iteration", "value"])
    frame["value"] = frame["value"].astype("float32")  # guard against pandas inferring int incorrectly

    frame.loc[
        frame["value"].isna() | (np.isfinite(frame["value"]) & (frame["value"] > MAX_ABS_METRIC_VALUE)), "value"
    ] = np.inf
    frame.loc[
        frame["value"].isna() | (np.isfinite(frame["value"]) & (frame["value"] < -MAX_ABS_METRIC_VALUE)), "value"
    ] = -np.inf

    return frame.astype(
        {"trial": "uint16", "agent": "uint16", "iteration": "uint32", "value": "float32"}
    )

def _all_episodes(rl_agent_views: dict[RLAlgorithm, list[list[RLAgentMetricsView]]]) -> list[int]:
    """Find all the episodes that were reached in at least one trial by at least one algorithm."""
    episodes: list[int] = []
    for rl_agent_views_by_trial in rl_agent_views.values():
        for rl_agent_view in rl_agent_views_by_trial:
            episodes += set.union(*(set(range(len(a.episode_return_history))) for a in rl_agent_view))

    return sorted(set(episodes))
