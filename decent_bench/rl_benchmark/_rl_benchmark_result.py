from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from decent_bench.rl_agents import RLAgent
from decent_bench.rl_algorithms import RLAlgorithm
from decent_bench.rl_benchmark._rl_benchmark_problem import RLBenchmarkProblem


@dataclass
class RLBenchmarkResult:
    """
    Result of a RL benchmark execution, containing the results and metadata.

    This class is used to store the results and metadata of a RL benchmark execution.
    It is returned by the :func:`~decent_bench.rl_benchmark.benchmark` function and contains
    all the information about the benchmark run, including the problem definition,
    algorithm states, table results, and plot results.

    * `problem`: contains the definition of the benchmark problem that was executed.
    * `result`: contains the final results of the algorithms after execution, organized by algorithm where
      each algorithm maps to a sequence of agent states list (one per trial) and mean episode returns.

    These results can be used to compute metrics after the benchmark run using
    :func:`~decent_bench.rl_benchmark.compute_metrics`.
    """

    problem: RLBenchmarkProblem
    result: Mapping[RLAlgorithm, Sequence[tuple[list[RLAgent], list[float]]]]
