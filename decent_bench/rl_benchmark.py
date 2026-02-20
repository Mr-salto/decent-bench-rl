import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from logging.handlers import QueueListener
from multiprocessing import Manager
from typing import TYPE_CHECKING, Literal, Callable, Any
import numpy as np
from rich.status import Status

from decent_bench.rl_benchmark_problem import RLBenchmarkProblem
from decent_bench.rl_algorithms import RLAlgorithm
from decent_bench.rl_agents import RLAgent, LinearDecreasingEpsilon
from decent_bench.environments import PettingZooEnv
import decent_bench.utils.interoperability as iop
from decent_bench.utils_rl.q_networks import QNetwork
from decent_bench.utils_rl.plot_return import plot_mean_episode_return
from decent_bench.schemes import AlwaysActive


from decent_bench.utils import logger
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.progress_bar import ProgressBarController

if TYPE_CHECKING:
    from decent_bench.utils.progress_bar import ProgressBarHandle

HIDDEN_SIZES = (64, 64)
DEVICE = "cpu"


def benchmark(
    algorithms: list[RLAlgorithm],
    benchmark_problem: RLBenchmarkProblem,
    n_trials: int = 1,
    max_processes: int | None = 1,

) -> None:
    """
    Benchmark MARL algorithms.

    Args:
        algorithms: MARL algorithms to benchmark
        benchmark_problem: problem to benchmark on, defines the environment.
        n_trials: number of times to run each algorithm on the benchmark problem, running more trials improves the
            statistical results, at least 30 trials are recommended for the central limit theorem to apply
    """
    _run_trials(algorithms, n_trials, benchmark_problem.n_agents, benchmark_problem.env_factory, max_processes)


def _run_trials( 
    algorithms: list[RLAlgorithm],
    n_trials: int,
    n_agents: int,
    env_factory: Callable[..., Any],
    max_processes: int | None,
    #progress_bar_ctrl: ProgressBarController,
    log_listener: QueueListener = None
) -> dict[RLAlgorithm, list[list[RLAgent]]]:
    #progress_bar_handle = progress_bar_ctrl.get_handle()
    if max_processes == 1:
        result = {
            alg: [_run_trial(alg, env_factory ,n_agents, trial) for trial in range(n_trials)]
            for alg in algorithms
        }
    return result


def _run_trial(
    algorithm: RLAlgorithm,
    env_factory: Callable[..., Any],
    n_agents: int,
    #progress_bar_handle: "ProgressBarHandle",
    trial: int
) -> None:
    #progress_bar_handle.start_progress_bar(algorithm, trial)
    alg = deepcopy(algorithm)

    agents = [RLAgent(i, action_space=None, observation_space=None, activation=AlwaysActive) for i in range(n_agents)]
    env = PettingZooEnv(agents=agents, env_factory=env_factory)

    for agent in agents:
        env_name = env.agent_to_env_name[agent]
        agent.action_space = env.action_spaces[env_name]
        agent.observation_space = env.observation_spaces[env_name]
        agent.n_actions = agent.action_space.n
        agent.epsilon_schedule = LinearDecreasingEpsilon(value=0)
        agent.epsilon = agent.epsilon_schedule(step=0)

        obs_shape = agent.observation_space.shape
        obs_dim = int(np.prod(obs_shape))
        qnet = QNetwork(obs_dim=obs_dim, n_actions=agent.n_actions, hidden_sizes=HIDDEN_SIZES, device=DEVICE)
        agent.q_network = qnet
    

    with warnings.catch_warnings(action="error"):
        try:
            alg.run(agents, env)
        except Exception as e:
            LOGGER.exception(f"An error or warning occurred when running {alg.name}: {type(e).__name__}: {e}")
    return agents
