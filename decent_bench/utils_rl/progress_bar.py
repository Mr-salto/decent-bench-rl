from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.managers import SyncManager
from typing import TYPE_CHECKING, Any, cast

from decent_bench.utils.progress_bar import ProgressBarController, ProgressBarHandle

if TYPE_CHECKING:
    from decent_bench.rl_algorithms import RLAlgorithm


@dataclass(eq=False)
class RLAlgorithmProgressAdapter:
    """Adapter exposing RL algorithm episodes as generic iteration count."""

    name: str
    iterations: int


@dataclass
class RLProgressBarHandle:
    """Picklable RL progress handle delegating to the decent-bench progress handle."""

    _base_handle: ProgressBarHandle
    _adapter_by_algorithm: dict[RLAlgorithm, RLAlgorithmProgressAdapter]

    def _get_adapter(self, algorithm: RLAlgorithm) -> RLAlgorithmProgressAdapter:
        adapter = self._adapter_by_algorithm[algorithm]
        if adapter is None:
            raise KeyError(f"No progress adapter registered for algorithm {algorithm.name}")
        return adapter

    def start_progress_bar(self, algorithm: RLAlgorithm, trial: int, initial_episode: int) -> None:
        """Start progress tracking for a trial from an initial episode index."""
        adapter = self._get_adapter(algorithm)
        self._base_handle.start_progress_bar(cast("Any", adapter), trial, initial_episode)

    def advance_progress_bar(self, algorithm: RLAlgorithm, episode: int) -> None:
        """Advance progress tracking based on the completed episode index."""
        adapter = self._get_adapter(algorithm)
        self._base_handle.advance_progress_bar(cast("Any", adapter), episode)


class RLProgressBarController:
    """Controller that maps RL episode progress onto the base progress infrastructure."""

    def __init__(  # noqa: PLR0917
        self,
        manager: SyncManager | None,
        algorithms: Sequence[RLAlgorithm],
        n_trials: int,
        progress_step: int | None,
        show_speed: bool = False,
        show_trial: bool = False,
    ) -> None:
        self._adapter_by_algorithm: dict[RLAlgorithm, RLAlgorithmProgressAdapter] = {
            alg: RLAlgorithmProgressAdapter(name=alg.name, iterations=alg.episodes) for alg in algorithms
        }
        base_algorithms = list(self._adapter_by_algorithm.values())
        self._base_controller = ProgressBarController(
            manager=manager,
            algorithms=cast("Sequence[Any]", base_algorithms),
            n_trials=n_trials,
            progress_step=progress_step,
            show_speed=show_speed,
            show_trial=show_trial,
        )
        self._handle = RLProgressBarHandle(
            _base_handle=self._base_controller.get_handle(),
            _adapter_by_algorithm=self._adapter_by_algorithm,
        )

    def mark_one_trial_as_complete(self, algorithm: RLAlgorithm, trial: int) -> None:
        """
        Mark one trial as completed for the given RL algorithm.

        Raises:
            KeyError: If no progress adapter is registered for the algorithm.

        """
        adapter = self._adapter_by_algorithm.get(algorithm)
        if adapter is None:
            raise KeyError(f"No progress adapter registered for algorithm {algorithm.name}")
        self._base_controller.mark_one_trial_as_complete(cast("Any", adapter), trial)

    def get_handle(self) -> RLProgressBarHandle:
        """Get a picklable handle to update progress from worker processes."""
        return self._handle

    def stop(self) -> None:
        """Stop the underlying progress listener and renderer."""
        self._base_controller.stop()
