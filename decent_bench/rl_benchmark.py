import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from decent_bench.environments import PettingZooEnv
from decent_bench.rl_agents import RLAgent
from decent_bench.rl_algorithms import RLAlgorithm
from decent_bench.rl_benchmark_problem import RLBenchmarkProblem
from decent_bench.schemes import AlwaysActive
from decent_bench.utils.logger import LOGGER
from decent_bench.utils_rl.plot_return import plot_benchmark_mean_episode_returns

HIDDEN_SIZES = (64, 64)
DEVICE = "cpu"


def benchmark(
    algorithms: list[RLAlgorithm],
    benchmark_problem: RLBenchmarkProblem,
    n_trials: int = 1,
    max_processes: int | None = 1,
) -> dict[RLAlgorithm, list[tuple[list[RLAgent], list[float]]]]:
    """
    Benchmark MARL algorithms.

    Args:
        algorithms: MARL algorithms to benchmark
        benchmark_problem: problem to benchmark on, defines the environment
        n_trials: number of times to run each algorithm on the benchmark problem, running more trials improves the
            statistical results, at least 30 trials are recommended for the central limit theorem to apply
        max_processes: maximum number of parallel processes to use for running trials, set to None to use default

    """
    result = _run_trials(algorithms, n_trials, benchmark_problem.n_agents, benchmark_problem.env_factory, max_processes)

    returns_by_algorithm = {alg.name: [trial_returns for _, trial_returns in trials] for alg, trials in result.items()}
    plot_benchmark_mean_episode_returns(returns_by_algorithm)
    return result


def _run_trials(
    algorithms: list[RLAlgorithm],
    n_trials: int,
    n_agents: int,
    env_factory: Callable[..., Any],
    max_processes: int | None,
) -> dict[RLAlgorithm, list[tuple[list[RLAgent], list[float]]]]:

    if max_processes == 1:
        result = {alg: [_run_trial(alg, env_factory, n_agents) for trial in range(n_trials)] for alg in algorithms}
    return result


def _run_trial(
    algorithm: RLAlgorithm,
    env_factory: Callable[..., Any],
    n_agents: int,
) -> tuple[list[RLAgent], list[float]]:

    alg = deepcopy(algorithm)

    agents = [RLAgent(i, action_space=None, observation_space=None, activation=AlwaysActive) for i in range(n_agents)]
    env = PettingZooEnv(agents=agents, env_factory=env_factory)

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
    return agents, mean_episode_returns
