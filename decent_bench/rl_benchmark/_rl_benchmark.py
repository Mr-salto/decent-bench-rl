import logging
import random
import warnings
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from json import JSONDecodeError
from logging.handlers import QueueListener
from multiprocessing import get_context
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import decent_bench.utils.interoperability as iop
from decent_bench.environments import PettingZooEnv
from decent_bench.rl_agents import RLAgent
from decent_bench.rl_algorithms import RLAlgorithm
from decent_bench.rl_benchmark._rl_benchmark_problem import (
    RLBenchmarkProblem,
    create_simple_adversary_problem,
    create_simple_spread_problem,
)
from decent_bench.rl_benchmark._rl_benchmark_result import RLBenchmarkResult
from decent_bench.schemes import AlwaysActive
from decent_bench.utils import logger
from decent_bench.utils.interoperability._rng import _set_seed
from decent_bench.utils.logger import LOGGER
from decent_bench.utils_rl.plot_return import plot_benchmark_mean_episode_returns
from decent_bench.utils_rl.progress_bar import RLProgressBarController
from decent_bench.utils_rl.runtime_metrics import (
    prepare_trial_runtime_metrics,
    start_runtime_metric_plotter,
    update_trial_runtime_metrics,
)

if TYPE_CHECKING:
    import queue
    from multiprocessing.context import SpawnContext
    from multiprocessing.managers import SyncManager

    from decent_bench.metrics import RuntimeMetric
    from decent_bench.utils_rl.progress_bar import RLProgressBarHandle
    from decent_bench.utils_rl.rl_checkpoint_manager import RLCheckpointManager


_SUPPORTED_MP_ENV_KINDS = {"simple_spread", "simple_adversary"}


def resume_benchmark(
    checkpoint_manager: "RLCheckpointManager",
    increase_trials: int = 0,
    create_backup: bool = True,
    *,
    max_processes: int | None = 1,
    progress_step: int | None = 1,
    show_speed: bool = False,
    show_trial: bool = False,
    runtime_metrics: "list[RuntimeMetric] | None" = None,
    log_level: int = logging.INFO,
) -> RLBenchmarkResult:
    """
    Resume an RL benchmark from an existing checkpoint directory.

    Args:
        checkpoint_manager: RLCheckpointManager instance to load checkpoints from.
        increase_trials: number of additional trials to run for each algorithm.
        create_backup: whether to create a backup of checkpoint directory before resuming.
        max_processes: maximum number of parallel processes to use for running trials.
        progress_step: if provided, progress updates every ``progress_step`` episodes.
            If ``None``, progress is updated once per trial at final episode.
        show_speed: whether to show speed in the progress bar.
        show_trial: whether to show active trial information in the progress bar.
        runtime_metrics: optional list of runtime metrics to compute and plot while training.
        log_level: minimum level to log, e.g. :data:`logging.INFO`.

    Returns:
        RLBenchmarkResult containing resumed results.

    Raises:
        ValueError: if checkpoint directory is invalid or increase_trials is negative.

    """
    if not checkpoint_manager.checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory '{checkpoint_manager.checkpoint_dir}' does not exist for resume")
    if checkpoint_manager.is_empty():
        raise ValueError(f"Checkpoint directory '{checkpoint_manager.checkpoint_dir}' is empty or invalid for resume")
    if increase_trials < 0:
        raise ValueError("increase_trials must be a non-negative integer")

    try:
        metadata = checkpoint_manager.load_metadata()
        if metadata is None or "n_trials" not in metadata:
            raise ValueError("Invalid or missing metadata in checkpoint directory")

        algorithms = checkpoint_manager.load_initial_algorithms()
        if algorithms is None:
            raise ValueError("Initial algorithms not found in checkpoint metadata")

        problem = checkpoint_manager.load_benchmark_problem()
        if problem is None:
            raise ValueError("Benchmark problem not found in checkpoint metadata")

    except (FileNotFoundError, KeyError) as e:
        raise ValueError(f"Invalid checkpoint directory: missing or corrupted metadata - {e}") from e
    except JSONDecodeError as e:
        raise ValueError(f"Invalid checkpoint directory: metadata is not valid JSON - {e}") from e

    if create_backup:
        checkpoint_manager.create_backup()

    log_listener, manager, mp_context = _init_logging_and_multiprocessing(log_level, max_processes)

    LOGGER.info(
        f"Resuming RL benchmark from checkpoint '{checkpoint_manager.checkpoint_dir}' with "
        f"{metadata['n_trials']} trials and algorithms: {[alg.name for alg in algorithms]}"
    )

    total_increase_trials = increase_trials + metadata.get("benchmark_metadata", {}).get("increased_trials", 0)
    n_trials = metadata["n_trials"] + total_increase_trials
    if increase_trials != 0:
        checkpoint_manager.append_metadata({"increased_trials": total_increase_trials})
        LOGGER.info(
            f"Increasing number of trials for each algorithm by {increase_trials}, "
            f"total increase is {total_increase_trials}"
        )

    LOGGER.info("Resuming benchmark execution")
    LOGGER.debug(f"Nr of agents: {problem.n_agents}")

    if runtime_metrics is not None and len(runtime_metrics) == 0:
        runtime_metrics = None

    result = _run_trials(
        algorithms,
        n_trials,
        problem=problem,
        manager=manager,
        max_processes=max_processes,
        progress_step=progress_step,
        show_speed=show_speed,
        show_trial=show_trial,
        log_listener=log_listener,
        mp_context=mp_context,
        checkpoint_manager=checkpoint_manager,
        runtime_metrics=runtime_metrics,
    )
    if log_listener is not None:
        log_listener.stop()

    returns_by_algorithm = {alg.name: [trial_returns for _, trial_returns in trials] for alg, trials in result.items()}
    plot_benchmark_mean_episode_returns(returns_by_algorithm)
    LOGGER.info("RL benchmark execution complete")
    return RLBenchmarkResult(problem=problem, result=result)


def benchmark(
    algorithms: list[RLAlgorithm],
    benchmark_problem: RLBenchmarkProblem,
    *,
    n_trials: int = 1,
    max_processes: int | None = 1,
    progress_step: int | None = 1,
    show_speed: bool = True,
    show_trial: bool = True,
    checkpoint_manager: "RLCheckpointManager | None" = None,
    runtime_metrics: "list[RuntimeMetric] | None" = None,
    log_level: int = logging.INFO,
) -> RLBenchmarkResult:
    """
    Benchmark MARL algorithms.

    Args:
        algorithms: MARL algorithms to benchmark
        benchmark_problem: problem to benchmark on, defines the environment
        n_trials: number of times to run each algorithm on the benchmark problem, running more trials improves the
            statistical results, at least 30 trials are recommended for the central limit theorem to apply
        max_processes: maximum number of parallel processes to use for running trials, set to None to use default
        progress_step: if provided, progress updates every ``progress_step`` episodes.
            If ``None``, progress is updated once per trial at final episode.
        show_speed: whether to show speed in the progress bar.
        show_trial: whether to show active trial information in the progress bar.
        checkpoint_manager: if provided, saves and loads trial-level checkpoints during benchmark execution
        runtime_metrics: optional list of runtime metrics to compute and plot while training.
        log_level: minimum level to log, e.g. :data:`logging.INFO`.

    Raises:
        ValueError: If the checkpoint directory is not empty when initializing the CheckpointManager.


    """
    log_listener, manager, mp_context = _init_logging_and_multiprocessing(log_level, max_processes)

    LOGGER.info("Starting benchmark execution")
    LOGGER.debug(f"Nr of agents: {benchmark_problem.n_agents}")

    if runtime_metrics is not None and len(runtime_metrics) == 0:
        runtime_metrics = None

    if checkpoint_manager is not None:
        if not checkpoint_manager.is_empty():
            raise ValueError(
                f"Checkpoint directory '{checkpoint_manager.checkpoint_dir}' is not empty. "
                "Please provide an empty or non-existent directory to save checkpoints."
            )
        checkpoint_manager.initialize(algorithms=algorithms, problem=benchmark_problem, n_trials=n_trials)
    else:
        LOGGER.info(
            "No checkpoint manager provided, running benchmark without checkpointing. "
            "Progress cannot be resumed if interrupted."
        )

    result = _run_trials(
        algorithms,
        n_trials,
        problem=benchmark_problem,
        manager=manager,
        max_processes=max_processes,
        progress_step=progress_step,
        show_speed=show_speed,
        show_trial=show_trial,
        log_listener=log_listener,
        mp_context=mp_context,
        checkpoint_manager=checkpoint_manager,
        runtime_metrics=runtime_metrics,
    )
    if log_listener is not None:
        log_listener.stop()

    returns_by_algorithm = {alg.name: [trial_returns for _, trial_returns in trials] for alg, trials in result.items()}
    plot_benchmark_mean_episode_returns(returns_by_algorithm)
    LOGGER.info("RL benchmark execution complete")
    return RLBenchmarkResult(problem=benchmark_problem, result=result)


def _run_trials(  # noqa: PLR0917
    algorithms: list[RLAlgorithm],
    n_trials: int,
    problem: RLBenchmarkProblem,
    manager: "SyncManager | None",
    max_processes: int | None,
    progress_step: int | None,
    show_speed: bool,
    show_trial: bool,
    log_listener: QueueListener | None = None,
    mp_context: "SpawnContext | None" = None,
    checkpoint_manager: "RLCheckpointManager | None" = None,
    runtime_metrics: "list[RuntimeMetric] | None" = None,
) -> dict[RLAlgorithm, list[tuple[list[RLAgent], list[float]]]]:
    results: dict[RLAlgorithm, list[tuple[list[RLAgent], list[float]]]] = defaultdict(list)

    runtime_plotter, runtime_plotter_queue = start_runtime_metric_plotter(runtime_metrics, mp_context)

    prog_ctrl = RLProgressBarController(manager, algorithms, n_trials, progress_step, show_speed, show_trial)
    progress_bar_handle = prog_ctrl.get_handle()

    to_run: dict[RLAlgorithm, list[int]] = defaultdict(list)
    if checkpoint_manager is not None:
        for alg_idx, alg in enumerate(algorithms):
            completed_trials = checkpoint_manager.get_completed_trials(alg_idx, n_trials)
            incompleted_trials = [t for t in range(n_trials) if t not in completed_trials]
            if len(incompleted_trials) > 0:
                to_run[alg] = incompleted_trials

            for trial in completed_trials:
                _, loaded_agents, loaded_returns = checkpoint_manager.load_trial_result(alg_idx, trial)
                results[alg].append((loaded_agents, loaded_returns))
                prog_ctrl.mark_one_trial_as_complete(alg, trial)
                LOGGER.debug(f"Loaded completed trial {trial} for algorithm {alg.name} from checkpoint")
    else:
        to_run = {alg: list(range(n_trials)) for alg in algorithms}

    LOGGER.debug(
        f"Trials to run: { {alg.name: trials for alg, trials in to_run.items()} }, "
        f"Trials completed: { {alg.name: len(results[alg]) for alg in algorithms} }"
    )

    if len(to_run) == 0:
        LOGGER.info("No trials are left to run!")
        return dict(results)

    if max_processes != 1 and problem.env_kind not in _SUPPORTED_MP_ENV_KINDS:
        raise ValueError(
            "Multiprocessing in RL benchmark currently supports built-in environment kinds "
            "'simple_spread' and 'simple_adversary' only. "
            "For custom env_factory, set max_processes=1."
        )

    trial_args = {
        alg: [
            (
                alg,
                problem.env_factory if max_processes == 1 else None,
                problem.env_kind,
                problem.n_agents,
                trial,
                alg_idx,
                _derive_trial_seed(iop.get_seed(), alg_idx, trial),
                progress_bar_handle,
                checkpoint_manager,
                runtime_metrics,
                runtime_plotter_queue,
            )
            for trial in to_run[alg]
        ]
        for alg_idx, alg in enumerate(algorithms)
    }

    if max_processes == 1:
        partial_result = {alg: [_run_trial(*args) for args in trial_args[alg]] for alg in trial_args}
    else:
        if log_listener is None:
            raise RuntimeError(
                "Log listener must be initialized for multiprocessing to handle logs from worker processes"
            )

        with ProcessPoolExecutor(
            initializer=logger.start_queue_logger,
            initargs=(log_listener.queue,),
            max_workers=max_processes,
            mp_context=mp_context,
        ) as executor:
            all_futures = {alg: [executor.submit(_run_trial, *args) for args in trial_args[alg]] for alg in trial_args}
            partial_result = {alg: [f.result() for f in as_completed(futures)] for alg, futures in all_futures.items()}

    for alg in partial_result:
        sorted_trials = sorted(partial_result[alg], key=itemgetter(0))
        results[alg].extend([trial_result[1] for trial_result in sorted_trials])

    # Clean up runtime plotter process
    if runtime_plotter is not None:
        runtime_plotter.shutdown()

    return dict(results)


def _run_trial(  # noqa: PLR0917
    algorithm: RLAlgorithm,
    env_factory: Callable[..., Any] | None,
    env_kind: str,
    n_agents: int,
    trial: int,
    alg_idx: int,
    trial_seed: int,
    progress_bar_handle: "RLProgressBarHandle",
    checkpoint_manager: "RLCheckpointManager | None" = None,
    runtime_metrics: "list[RuntimeMetric] | None" = None,
    runtime_plotter_queue: "queue.Queue[Any] | None" = None,
) -> tuple[int, tuple[list[RLAgent], list[float]]]:

    _set_seed(trial_seed, set_global_seed=False)

    alg = deepcopy(algorithm)

    trial_env_factory = env_factory if env_factory is not None else _get_env_factory(env_kind, n_agents)

    agents = [RLAgent(i, action_space=None, observation_space=None, activation=AlwaysActive) for i in range(n_agents)]
    env = PettingZooEnv(agents=agents, env_factory=trial_env_factory, max_cycles=alg.episode_length - 1)

    for agent in agents:
        env_name = env.agent_to_env_name[agent]
        agent.action_space = env.action_spaces[env_name]
        agent.observation_space = env.observation_spaces[env_name]

    progress_bar_handle.start_progress_bar(algorithm, trial, 0)

    trial_runtime_metrics = prepare_trial_runtime_metrics(runtime_metrics, alg.name, trial, runtime_plotter_queue)
    trial_problem = RLBenchmarkProblem(env_kind=env_kind, env_factory=trial_env_factory, n_agents=n_agents)

    def episode_callback(episode: int) -> None:
        progress_bar_handle.advance_progress_bar(algorithm, episode)
        update_trial_runtime_metrics(trial_runtime_metrics, trial_problem, agents, episode, alg.episodes)

    mean_episode_returns = []
    with warnings.catch_warnings(action="error"):
        try:
            mean_episode_returns = alg.run(agents, env, episode_callback=episode_callback)
            if checkpoint_manager is not None:
                checkpoint_manager.mark_trial_complete(
                    alg_idx=alg_idx,
                    trial=trial,
                    algorithm=alg,
                    agents=agents,
                    mean_episode_returns=mean_episode_returns,
                    rng_state=iop.get_rng_state(),
                )
        except Exception as e:
            LOGGER.exception(f"An error or warning occurred when running {alg.name}: {type(e).__name__}: {e}")

    return trial, (agents, mean_episode_returns)


def _init_logging_and_multiprocessing(
    log_level: int,
    max_processes: int | None,
) -> tuple[QueueListener | None, "SyncManager | None", "SpawnContext | None"]:
    if max_processes == 1:
        logger.start_logger(log_level)
        return None, None, None

    mp_context = get_context("spawn")
    try:
        manager = mp_context.Manager()
    except RuntimeError as e:
        if _is_multiprocessing_main_guard_error(e):
            raise RuntimeError(
                "Failed to start multiprocessing workers. Benchmark execution "
                "must be launched inside a guarded main entrypoint. Wrap your benchmark call in:\n\n"
                "if __name__ == '__main__':\n"
                "    ... call decent_bench.rl_benchmark.benchmark(...)\n\n"
                "This prevents child processes from re-running top-level script code during import."
            ) from e
        raise

    log_listener = logger.start_log_listener(manager, log_level)
    LOGGER.debug("Using spawn multiprocessing context for RL benchmark")
    return log_listener, manager, mp_context


def _is_multiprocessing_main_guard_error(exc: RuntimeError) -> bool:
    """Return True for the common spawn bootstrap error caused by missing main guard."""
    msg = str(exc)
    return "start a new process before the" in msg and "bootstrapping phase" in msg


def _derive_trial_seed(base_seed: int | None, algorithm_index: int, trial: int) -> int:
    """Derive a deterministic per-trial seed from a base seed."""
    if base_seed is None:
        base_seed = random.randint(0, 2**32 - 1)

    return int((base_seed + 0x9E3779B9 * (algorithm_index + 1) + 0x85EBCA6B * (trial + 1)) % (2**32))


def _get_env_factory(env_kind: str, n_agents: int) -> Callable[..., Any]:
    if env_kind == "simple_spread":
        return create_simple_spread_problem(n_agents=n_agents).env_factory
    if env_kind == "simple_adversary":
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1 for simple_adversary, got {n_agents}")
        return create_simple_adversary_problem(n_good_agents=n_agents - 1).env_factory

    raise ValueError("Unsupported env_kind for multiprocessing. Supported kinds: 'simple_spread', 'simple_adversary'.")
