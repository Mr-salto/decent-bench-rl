import json
import pickle  # noqa: S403
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import decent_bench.utils.interoperability as iop
from decent_bench.benchmark import MetricResult
from decent_bench.rl_agents import RLAgent
from decent_bench.rl_algorithms import RLAlgorithm
from decent_bench.rl_benchmark import (
    RLBenchmarkProblem,
    RLBenchmarkResult,
)
from decent_bench.utils.logger import LOGGER


class RLCheckpointManager:
    """
    Manages checkpoint directory structure and file operations for RL benchmark execution.

    The CheckpointManager creates and maintains a hierarchical directory structure for storing
    checkpoint data during benchmark execution. This allows benchmarks to be resumed if interrupted,
    and provides incremental saving of results as trials complete.

    Directory Structure:
        The checkpoint directory is organized as follows::

            checkpoint_dir/
            ├── metadata.json                   # Run configuration and algorithm metadata
            ├── benchmark_problem.pkl           # Initial benchmark problem state (before any trials)
            ├── initial_algorithms.pkl          # Initial algorithm states (before any trials)
            ├── metric_computation.pkl          # Computed metrics results (after all trials complete)
            ├── algorithm_0/                    # Directory for first algorithm
            │   ├── trial_0/                    # Directory for trial 0
            │   │   ├── checkpoint_0000100.pkl  # Combined algorithm+network state at iteration 100
            │   │   ├── checkpoint_0000200.pkl  # Combined algorithm+network state at iteration 200
            │   │   ├── progress.json           # {"last_completed_iteration": N}
            │   │   └── complete.json           # Marker file, contains path to final checkpoint
            │   ├── trial_1/
            │   │   └── ...
            │   └── trial_N/
            │       └── ...
            └── results/                        # Results directory for storing final tables and plots after completion
                ├── plots_fig1.png              # Final plot for figure 1 with plot results
                ├── plots_fig2.png              # Final plot for figure 2 with plot results
                ├── table.tex                   # Final LaTeX file with table results
                └── table.txt                   # Final text file with table results

    File Descriptions:
        - **metadata.json**: Benchmark configuration and any user-provided metadata
          (e.g., hyperparameters, system info). User-provided metadata can be added through the
          :func:`~decent_bench.benchmark.benchmark` function or appended later using
          :func:`~decent_bench.utils.checkpoint_manager.CheckpointManager.append_metadata`.
        - **benchmark_problem.pkl**: Initial benchmark problem state before any trials run.
        - **initial_algorithms.pkl**: Initial algorithm states before any trials run.
        - **metric_computation.pkl**: Computed metrics results after :func:`~decent_bench.benchmark.compute_metrics`
          completes.
        - **checkpoint_NNNNNNN.pkl**: Combined checkpoint containing both algorithm and network state.
          This preserves shared object references and ensures consistency between algorithm and network
          states at each checkpoint. The checkpoint data is a dictionary with the following structure:

            - algorithm: :class:`~decent_bench.algorithms.Algorithm`
            - network: :class:`~decent_bench.networks.Network`
            - iteration: iteration

          where "algorithm" is the :class:`~decent_bench.algorithms.Algorithm` object with its internal
          state at the checkpoint, "network" is the :class:`~decent_bench.networks.Network` object with agent states
          at the checkpoint and "iteration" is the iteration number of the checkpoint.
        - **progress.json**: Tracks the last completed iteration within a trial.
        - **complete.json**: Marker file, contains path to final checkpoint.
        - **plots_figX.png**: Final plots for figures after benchmark completion.
        - **table.tex**: Final LaTeX file with table results after benchmark completion.
        - **table.txt**: Final text file with table results after benchmark completion.

    Thread Safety:
        - Each trial writes to its own directory, avoiding write conflicts.
        - Completed trial results are loaded read-only.
        - Metadata is written once at initialization.

    Args:
        checkpoint_dir: Path to the checkpoint directory.
        keep_n_checkpoints: Maximum number of iteration checkpoints to keep per trial.
            Older checkpoints are automatically deleted to save disk space.

    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        keep_n_checkpoints: int = 1,
        benchmark_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize RLCheckpointManager with a checkpoint directory path.

        Args:
            checkpoint_dir: Path to save checkpoints during execution. If provided, progress will be saved
                at regular intervals allowing resumption if interrupted. When starting a new benchmark
                the directory must be empty or non-existent.
            keep_n_checkpoints: Maximum number of iteration checkpoints to keep per trial.
                Kept for future compatibility.
                Older checkpoints are automatically deleted to save disk space.
            benchmark_metadata: Optional dictionary of additional metadata to save in the checkpoint directory,
                    such as hyperparameters or system information. This can be useful for keeping track of the benchmark
                    configuration and context when analyzing results later.

        Raises:
            ValueError: If keep_n_checkpoints is not a positive integer.

        """
        if keep_n_checkpoints <= 0:
            raise ValueError(f"keep_n_checkpoints must be a positive integer, got {keep_n_checkpoints}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_n_checkpoints = keep_n_checkpoints
        self._metadata = benchmark_metadata
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def is_empty(self) -> bool:
        """Check if checkpoint directory is empty or doesn't exist."""
        if not self.checkpoint_dir.exists():
            return True
        return not any(self.checkpoint_dir.iterdir())

    def initialize(
        self,
        algorithms: list[RLAlgorithm],
        problem: RLBenchmarkProblem,
        n_trials: int,
    ) -> None:
        """
        Initialize checkpoint directory structure for a new benchmark run.

        Args:
            algorithms: List of RLAlgorithm objects to be benchmarked.
            problem: RLBenchmarkProblem configuration for the benchmark.
            n_trials: Total number of trials to run for each algorithm, used for resuming.

        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata: dict[str, Any] = {
            "n_trials": n_trials,
            "algorithms": [
                {
                    "name": alg.name,
                    "episodes": alg.episodes,
                    "index": idx,
                }
                for idx, alg in enumerate(algorithms)
            ],
        }
        if self._metadata is not None:
            metadata["benchmark_metadata"] = self._metadata
        if iop.get_seed() is not None:
            metadata["rng_seed"] = iop.get_seed()

        # Save initial state and metadata for resuming later if needed
        self._save_metadata(metadata)
        self._save_initial_algorithms(algorithms)
        self._save_benchmark_problem(problem)

        # Create algorithm directories
        for idx in range(len(algorithms)):
            self._get_algorithm_dir(idx).mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Initialized checkpoint directory at '{self.checkpoint_dir}'")

    def create_backup(self) -> Path:
        """
        Create a backup of the existing checkpoint directory.

        Returns:
            Path to the created backup zip file.

        Raises:
            FileExistsError: If the backup file already exists.

        """
        backup_path = Path(f"{self.checkpoint_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")  # noqa: DTZ005
        if backup_path.exists():
            raise FileExistsError(f"Backup file '{backup_path}' already exists")

        shutil.make_archive(str(backup_path.with_suffix("")), "zip", self.checkpoint_dir)
        LOGGER.info(f"Created backup of checkpoint directory at '{backup_path}'")
        return backup_path

    def append_metadata(self, additional_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Append additional metadata to existing checkpoint metadata.

        This can be used to add information after initialization, such as system resource usage,
        hyperparameters, or other contextual information that may be relevant for analyzing results later.

        Args:
            additional_metadata: Dictionary of additional metadata to append to the existing metadata.

        Returns:
            Updated metadata dictionary after appending the additional metadata.

        """
        metadata = self.load_metadata()
        if "benchmark_metadata" not in metadata:
            metadata["benchmark_metadata"] = {}
        metadata["benchmark_metadata"].update(additional_metadata)
        self._save_metadata(metadata)
        return metadata

    def load_initial_algorithms(self) -> list[RLAlgorithm]:
        """
        Load initial RL algorithm states from checkpoint.

        Returns:
            List of RLAlgorithm objects representing the initial algorithm states.

        """
        initial_path = self.checkpoint_dir / "initial_algorithms.pkl"
        with initial_path.open("rb") as f:
            ret: list[RLAlgorithm] = pickle.load(f)  # noqa: S301
        return ret

    def load_benchmark_problem(self) -> RLBenchmarkProblem:
        """
        Load benchmark problem configuration from checkpoint.

        Returns:
            BenchmarkProblem object representing the benchmark problem configuration.

        Raises:
            ValueError: If the benchmark problem payload is invalid.

        """
        problem_path = self.checkpoint_dir / "benchmark_problem.pkl"
        with problem_path.open("rb") as f:
            payload = pickle.load(f)  # noqa: S301

        required_keys = ("env_kind", "env_config", "agents")
        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise ValueError(f"Invalid benchmark problem payload: missing keys {missing_keys}")

        env_kind = payload["env_kind"]
        env_config = dict(payload["env_config"])
        agents = payload["agents"]

        return RLBenchmarkProblem(
            env_kind=env_kind,
            agents=agents,
            env_config=env_config,
        )

    def save_checkpoint(
        self,
        *,
        alg_idx: int,
        trial: int,
        algorithm: RLAlgorithm,
        agents: list[RLAgent],
        mean_episode_returns: list[float],
        rng_state: dict[str, Any],
    ) -> Path:
        """
        Save checkpoint for a specific algorithm trial.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).
            algorithm: Algorithm object with current internal state.
            agents: List of RLAgent objects with current agent states and metrics.
            mean_episode_returns: List of mean episode returns across agents.
            rng_state: RNG snapshot for deterministic resume.

        Returns:
            Path to the saved checkpoint file.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save both algorithm and agents in a single pickle file to preserve shared object references
        checkpoint_path = trial_dir / "checkpoint_final.pkl"
        checkpoint_data = {
            "algorithm": algorithm,
            "agents": agents,
            "mean_episode_returns": mean_episode_returns,
            "rng_state": rng_state,
        }
        with checkpoint_path.open("wb") as f:
            pickle.dump(checkpoint_data, f)

        LOGGER.debug(f"Saved checkpoint: alg={alg_idx}, trial={trial}")

        self._cleanup_old_checkpoints(alg_idx, trial)
        return checkpoint_path

    def load_checkpoint(
        self, alg_idx: int, trial: int
    ) -> tuple[RLAlgorithm, list[RLAgent], list[float], dict[str, Any]] | None:
        """
        Load the latest checkpoint for a specific algorithm trial.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            Tuple of (algorithm, agents, mean_episode_returns, rng_state) or None if no checkpoint exists.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)

        # Load both algorithm and network from single checkpoint file
        checkpoint_path = trial_dir / "checkpoint_final.pkl"

        if not checkpoint_path.exists():
            return None

        with checkpoint_path.open("rb") as f:
            checkpoint_data = pickle.load(f)  # noqa: S301

        algorithm: RLAlgorithm = checkpoint_data["algorithm"]
        agents: list[RLAgent] = checkpoint_data["agents"]
        mean_episode_returns: list[float] = checkpoint_data["mean_episode_returns"]
        rng_state: dict[str, Any] = checkpoint_data["rng_state"]

        LOGGER.debug(f"Loaded checkpoint: alg={alg_idx}, trial={trial}")
        return algorithm, agents, mean_episode_returns, rng_state

    def mark_trial_complete(
        self,
        *,
        alg_idx: int,
        trial: int,
        algorithm: RLAlgorithm,
        agents: list[RLAgent],
        mean_episode_returns: list[float],
        rng_state: dict[str, Any],
    ) -> Path:
        """
        Mark a trial as complete and save final result.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).
            algorithm: Final Algorithm state after all episodes complete.
            agents: List of RLAgent objects with final agent states and metrics.
            mean_episode_returns: List of mean episode returns across agents.
            rng_state: RNG snapshot for deterministic resume.

        Returns:
            Path to the saved final checkpoint file.

        """
        checkpoint_path = self.save_checkpoint(
            alg_idx=alg_idx,
            trial=trial,
            algorithm=algorithm,
            agents=agents,
            mean_episode_returns=mean_episode_returns,
            rng_state=rng_state,
        )

        # Mark as complete
        trial_dir = self._get_trial_dir(alg_idx, trial)
        complete_path = trial_dir / "complete.json"
        completed_metadata = {
            "alg_name": algorithm.name,
            "alg_idx": alg_idx,
            "trial": trial,
            "episodes": algorithm.episodes,
            "checkpoint_path": str(checkpoint_path),
        }
        with complete_path.open("w") as f:
            json.dump(completed_metadata, f)

        LOGGER.debug(f"Marked trial complete: alg={alg_idx}, trial={trial}")
        return checkpoint_path

    def unmark_trial_complete(self, alg_idx: int, trial: int) -> None:
        """
        Remove the completion marker for a trial, allowing it to be rerun.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        complete_path = trial_dir / "complete.json"
        if complete_path.exists():
            complete_path.unlink()
            LOGGER.debug(f"Unmarked trial complete: alg={alg_idx}, trial={trial}")

    def is_trial_complete(self, alg_idx: int, trial: int) -> bool:
        """
        Check if a trial has been completed.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            True if the trial has completed, False otherwise.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        return (trial_dir / "complete.json").exists()

    def load_trial_result(self, alg_idx: int, trial: int) -> tuple[RLAlgorithm, list[RLAgent], list[float]]:
        """
        Load final result of a completed trial.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            Tuple of (RLAlgorithm object, list of RLAgent objects, list of mean
            episode returns) with final state after all episodes.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        complete_path = trial_dir / "complete.json"

        with complete_path.open(encoding="utf-8") as f:
            completed_metadata = json.load(f)
        final_path = Path(completed_metadata["checkpoint_path"])

        with final_path.open("rb") as f:
            checkpoint_data = pickle.load(f)  # noqa: S301

        alg: RLAlgorithm = checkpoint_data["algorithm"]
        agents: list[RLAgent] = checkpoint_data["agents"]
        mean_episode_returns: list[float] = checkpoint_data["mean_episode_returns"]
        return alg, agents, mean_episode_returns

    def get_completed_trials(self, alg_idx: int, n_trials: int) -> list[int]:
        """
        Get list of completed trial numbers for an algorithm.

        Args:
            alg_idx: Algorithm index (0-based).
            n_trials: Total number of trials in the benchmark.

        Returns:
            List of completed trial numbers (0-based).

        """
        return [trial for trial in range(n_trials) if self.is_trial_complete(alg_idx, trial)]

    def load_metadata(self) -> dict[str, Any]:
        """
        Load checkpoint metadata.

        If no metadata file exists, returns an empty dictionary.

        Returns:
            Dictionary containing benchmark_metadata and algorithms list.

        """
        metadata_path = self.checkpoint_dir / "metadata.json"

        if not metadata_path.exists():
            return {}

        with metadata_path.open(encoding="utf-8") as f:
            metadata: dict[str, Any] = json.load(f)
        return metadata

    def load_benchmark_result(self) -> RLBenchmarkResult:
        """
        Load benchmark problem configuration and states from checkpoint.

        If an algorithm does not have all trials completed, its results will be skipped and not included in the loaded
        benchmark result. This is to ensure that the metrics are not skewed by incomplete data and only include
        algorithms with full results. A warning will be logged for any incomplete algorithms.

        Returns:
            RLBenchmarkResult object containing the loaded RL benchmark problem,
            initial RL algorithms, and initial RL agents states.

        """
        problem = self.load_benchmark_problem()
        algorithms = self.load_initial_algorithms()
        metadata = self.load_metadata()
        n_trials = metadata["n_trials"]
        result: dict[RLAlgorithm, list[tuple[list[RLAgent], list[float]]]] = {}
        for idx, alg in enumerate(algorithms):
            completed_trials = self.get_completed_trials(idx, n_trials)
            if len(completed_trials) != n_trials:
                LOGGER.warning(
                    f"Algorithm '{alg.name}' has {len(completed_trials)}/{n_trials} completed trials. "
                    "Results will not be loaded for this algorithm."
                )
                continue
            for trial in completed_trials:
                loaded_alg, loaded_agents, loaded_mean_episode_returns = self.load_trial_result(idx, trial)
                if loaded_alg.name != alg.name:
                    LOGGER.warning(
                        f"Algorithm mismatch in trial {trial} for algorithm {alg.name}, loaded {loaded_alg.name}. "
                        "Results will not be loaded for this algorithm."
                    )
                    result.pop(alg, None)  # Remove any previously loaded states for this algorithm
                    break
                if alg not in result:
                    result[alg] = []
                result[alg].append((loaded_agents, loaded_mean_episode_returns))

        return RLBenchmarkResult(
            problem=problem,
            result=result,
        )

    def save_metrics_result(self, metrics_result: MetricResult) -> None:
        """
        Save the computed metrics result to the checkpoint directory.

        Args:
            metrics_result: MetricsResult object containing the computed metrics to save.

        """
        metric_path = self.checkpoint_dir / "metric_computation.pkl"
        with metric_path.open("wb") as f:
            pickle.dump(metrics_result, f)
        LOGGER.info(f"Saved computed metrics result to {metric_path}")

    def load_metrics_result(self) -> MetricResult:
        """
        Load the computed metrics result from the checkpoint directory.

        Returns:
            MetricsResult object containing the computed metrics.

        """
        metric_path = self.checkpoint_dir / "metric_computation.pkl"
        with metric_path.open("rb") as f:
            metrics_result: MetricResult = pickle.load(f)  # noqa: S301
        LOGGER.info(f"Loaded computed metrics result from {metric_path}")
        return metrics_result

    def get_results_path(self) -> Path:
        """
        Get the path to the results directory within the checkpoint directory.

        Returns:
            Path to the results directory within the checkpoint directory.

        """
        return self.checkpoint_dir / "results"

    def clear(self) -> None:
        """
        Remove entire checkpoint directory and all its contents.

        Warning:
            This permanently deletes all checkpoint data.

        """
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            LOGGER.info(f"Cleared checkpoint directory: {self.checkpoint_dir}")

    def _get_trial_dir(self, alg_idx: int, trial: int) -> Path:
        """Get directory path for a specific trial."""
        return self._get_algorithm_dir(alg_idx) / f"trial_{trial}"

    def _get_algorithm_dir(self, alg_idx: int) -> Path:
        """Get directory path for an algorithm."""
        return self.checkpoint_dir / f"algorithm_{alg_idx}"

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save metadata to checkpoint directory."""
        metadata_path = self.checkpoint_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f)
        LOGGER.debug(f"Saved metadata to {metadata_path}")

    def _save_initial_algorithms(self, algorithms: list[RLAlgorithm]) -> None:
        """Save initial algorithm states before any trials run."""
        initial_path = self.checkpoint_dir / "initial_algorithms.pkl"
        with initial_path.open("wb") as f:
            pickle.dump(algorithms, f)
        LOGGER.debug(f"Saved initial algorithms to {initial_path}")

    def _save_benchmark_problem(self, problem: RLBenchmarkProblem) -> None:
        """Save benchmark problem configuration metadata for deterministic reconstruction."""
        problem_path = self.checkpoint_dir / "benchmark_problem.pkl"
        payload = {
            "env_kind": problem.env_kind,
            "env_config": problem.env_config,
            "agents": problem.agents,
        }
        with problem_path.open("wb") as f:
            pickle.dump(payload, f)
        LOGGER.debug(f"Saved benchmark problem to {problem_path}")

    def _cleanup_old_checkpoints(self, alg_idx: int, trial: int) -> None:
        """
        Remove old trial checkpoint files, keeping only the most recent N.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        if not trial_dir.exists():
            return

        # Find all iteration checkpoint files
        checkpoint_files = list(trial_dir.glob("checkpoint_*.pkl"))

        # Remove older checkpoints
        if len(checkpoint_files) > self.keep_n_checkpoints:
            for file_to_remove in checkpoint_files[self.keep_n_checkpoints :]:
                try:
                    file_to_remove.unlink()
                    LOGGER.debug(f"Removed old checkpoint: {file_to_remove}")
                except FileNotFoundError:
                    LOGGER.debug(f"Checkpoint file already removed by another process: {file_to_remove}")
