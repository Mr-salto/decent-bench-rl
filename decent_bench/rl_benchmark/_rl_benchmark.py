import logging
import warnings
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from json import JSONDecodeError
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import decent_bench.utils.interoperability as iop
from decent_bench.environments import PettingZooEnv
from decent_bench.rl_agents import RLAgent
from decent_bench.rl_algorithms import RLAlgorithm
from decent_bench.rl_benchmark._rl_benchmark_problem import RLBenchmarkProblem
from decent_bench.rl_benchmark._rl_benchmark_result import RLBenchmarkResult
from decent_bench.schemes import AlwaysActive
from decent_bench.utils import logger
from decent_bench.utils.logger import LOGGER
from decent_bench.utils_rl.plot_return import plot_benchmark_mean_episode_returns

if TYPE_CHECKING:
    from decent_bench.utils_rl.rl_checkpoint_manager import RLCheckpointManager


def resume_benchmark(
    checkpoint_manager: "RLCheckpointManager",
    increase_trials: int = 0,
    create_backup: bool = True,
    *,
    max_processes: int | None = 1,
    log_level: int = logging.INFO,
) -> RLBenchmarkResult:
    """
    Resume an RL benchmark from an existing checkpoint directory.

    Args:
        checkpoint_manager: RLCheckpointManager instance to load checkpoints from.
        increase_trials: number of additional trials to run for each algorithm.
        create_backup: whether to create a backup of checkpoint directory before resuming.
        max_processes: maximum number of parallel processes to use for running trials.
        log_level: minimum level to log, e.g. :data:`logging.INFO`.

    Returns:
        RLBenchmarkResult containing resumed results.

    Raises:
        ValueError: if checkpoint directory is invalid or increase_trials is negative.

    """
    logger.start_logger(log_level=log_level)

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

    result = _run_trials(
        algorithms,
        n_trials,
        n_agents=problem.n_agents,
        env_factory=problem.env_factory,
        max_processes=max_processes,
        checkpoint_manager=checkpoint_manager,
    )

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
    checkpoint_manager: "RLCheckpointManager | None" = None,
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
        checkpoint_manager: if provided, saves and loads trial-level checkpoints during benchmark execution
        log_level: minimum level to log, e.g. :data:`logging.INFO`.

    Raises:
        ValueError: If the checkpoint directory is not empty when initializing the CheckpointManager.


    """
    logger.start_logger(log_level=log_level)

    LOGGER.info("Starting benchmark execution")
    LOGGER.debug(f"Nr of agents: {benchmark_problem.n_agents}")
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
        n_agents=benchmark_problem.n_agents,
        env_factory=benchmark_problem.env_factory,
        max_processes=max_processes,
        checkpoint_manager=checkpoint_manager,
    )

    returns_by_algorithm = {alg.name: [trial_returns for _, trial_returns in trials] for alg, trials in result.items()}
    plot_benchmark_mean_episode_returns(returns_by_algorithm)
    LOGGER.info("RL benchmark execution complete")
    return RLBenchmarkResult(problem=benchmark_problem, result=result)


def _run_trials(  # noqa: PLR0917
    algorithms: list[RLAlgorithm],
    n_trials: int,
    n_agents: int,
    env_factory: Callable[..., Any],
    max_processes: int | None,
    checkpoint_manager: "RLCheckpointManager | None" = None,
) -> dict[RLAlgorithm, list[tuple[list[RLAgent], list[float]]]]:
    results: dict[RLAlgorithm, list[tuple[list[RLAgent], list[float]]]] = defaultdict(list)

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

    if max_processes != 1:
        raise NotImplementedError("RL benchmark currently supports max_processes == 1 only")

    for alg_idx, alg in enumerate(algorithms):
        partial_result: list[tuple[int, tuple[list[RLAgent], list[float]]]] = []
        for trial in to_run[alg]:
            LOGGER.debug(f"Running trial {trial} for algorithm {alg.name}")
            trained_alg, agents, mean_episode_returns = _run_trial(alg, env_factory, n_agents)
            partial_result.append((trial, (agents, mean_episode_returns)))

            if checkpoint_manager is not None:
                checkpoint_manager.mark_trial_complete(
                    alg_idx=alg_idx,
                    trial=trial,
                    algorithm=trained_alg,
                    agents=agents,
                    mean_episode_returns=mean_episode_returns,
                    rng_state=iop.get_rng_state(),
                )

        sorted_trials = sorted(partial_result, key=itemgetter(0))
        results[alg].extend([trial_result for _, trial_result in sorted_trials])

    return dict(results)


def _run_trial(
    algorithm: RLAlgorithm,
    env_factory: Callable[..., Any],
    n_agents: int,
) -> tuple[RLAlgorithm, list[RLAgent], list[float]]:

    alg = deepcopy(algorithm)

    agents = [RLAgent(i, action_space=None, observation_space=None, activation=AlwaysActive) for i in range(n_agents)]
    env = PettingZooEnv(agents=agents, env_factory=env_factory, max_cycles=alg.episode_length - 1)

    for agent in agents:
        env_name = env.agent_to_env_name[agent]
        agent.action_space = env.action_spaces[env_name]
        agent.observation_space = env.observation_spaces[env_name]

    mean_episode_returns = []
    with warnings.catch_warnings(action="error"):
        try:
            mean_episode_returns = alg.run(agents, env)
        except Exception as e:
            LOGGER.exception(f"An error or warning occurred when running {alg.name}: {type(e).__name__}: {e}")
    return alg, agents, mean_episode_returns
