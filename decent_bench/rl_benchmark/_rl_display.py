from pathlib import Path
from typing import TYPE_CHECKING, Literal

from decent_bench.benchmark._display import display_metrics as _display_metrics
from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.metrics import ComputationalCost, Metric

if TYPE_CHECKING:
    from decent_bench.utils.checkpoint_manager import CheckpointManager


def display_metrics(
    metrics_result: MetricResult | None = None,
    checkpoint_manager: "CheckpointManager | None" = None,
    *,
    table_metrics: list[Metric] | None = None,
    plot_metrics: list[Metric] | list[list[Metric]] | None = None,
    table_fmt: Literal["grid", "latex"] = "grid",
    plot_grid: bool = True,
    individual_plots: bool = False,
    computational_cost: ComputationalCost | None = None,
    x_axis_scaling: float = 1e-4,
    compare_iterations_and_computational_cost: bool = False,
    save_path: str | Path | None = None,
    plot_format: Literal["png", "pdf", "svg"] = "png",
) -> None:
    """Display RL metrics using the shared display pipeline."""
    _display_metrics(
        metrics_result=metrics_result,
        checkpoint_manager=checkpoint_manager,
        table_metrics=table_metrics,
        plot_metrics=plot_metrics,
        table_fmt=table_fmt,
        plot_grid=plot_grid,
        individual_plots=individual_plots,
        computational_cost=computational_cost,
        x_axis_scaling=x_axis_scaling,
        compare_iterations_and_computational_cost=compare_iterations_and_computational_cost,
        save_path=save_path,
        plot_format=plot_format,
    )
