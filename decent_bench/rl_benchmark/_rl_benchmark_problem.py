from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from mpe2 import simple_adversary_v3, simple_spread_v3

from decent_bench.environments import MPE
from decent_bench.rl_agents import RLAgent
from decent_bench.schemes import AlwaysActive

RENDER_MODE = "human"  # "human" | None


@dataclass(eq=False)
class RLBenchmarkProblem:
    """
    RL benchmark problem describing environment setup.

    The problem intentionally does not own training-loop settings such as
    episode length or number of episodes.

    Args:
        env_kind: environment identifier.
        n_agents: number of agents in the environment
        agents: template RL agents used to create per-trial deep copies.
        env_config: minimal configuration required to instantiate the underlying Parallel env.

    """

    env_kind: str
    n_agents: int
    agents: list[RLAgent]
    env_config: dict[str, Any]

    def create_env(self, agents: list[RLAgent] | None = None) -> MPE:
        """Instantiate a fresh environment wrapper using this problem metadata."""

        def env_factory(**_env_kwargs: Any) -> Any:  # noqa: ANN401
            if self.env_kind == "simple_spread":
                return simple_spread_v3.parallel_env(**self.env_config)
            if self.env_kind == "simple_adversary":
                return simple_adversary_v3.parallel_env(**self.env_config)
            raise ValueError(f"Unsupported env_kind: {self.env_kind}")

        trial_agents = deepcopy(self.agents) if agents is None else agents
        return MPE(agents=trial_agents, env_factory=env_factory)


def create_simple_spread_problem(
    n_agents: int = 1,
    *,
    render_mode: str | None = RENDER_MODE,
) -> RLBenchmarkProblem:
    """
    Create out-of-the-box simple-spread problems using MPE.

    Args:
        n_agents: number of agents in the environment
        render_mode: MPE render mode.

    """
    env_config = {
        "N": n_agents,
        "render_mode": render_mode,
    }
    agents = [RLAgent(i, activation=AlwaysActive()) for i in range(n_agents)]

    return RLBenchmarkProblem(
        env_kind="simple_spread",
        n_agents=n_agents,
        agents=agents,
        env_config=env_config,
    )


def create_simple_adversary_problem(
    n_good_agents: int = 2,
    *,
    render_mode: str | None = RENDER_MODE,
) -> RLBenchmarkProblem:
    """
    Create an out-of-the-box simple-adversary problem using MPE.

    Args:
        n_good_agents: number of cooperative (non-adversary) agents.
        render_mode: MPE render mode.

    """
    # simple_adversary_v3 is used with one adversary by design.
    total_agents = n_good_agents + 1
    env_config = {
        "N": n_good_agents,
        "render_mode": render_mode,
    }
    agents = [RLAgent(i, activation=AlwaysActive()) for i in range(total_agents)]

    return RLBenchmarkProblem(
        env_kind="simple_adversary",
        n_agents=total_agents,
        agents=agents,
        env_config=env_config,
    )


def create_mpe_problem(
    n_agents: int,
    env_kind: str = "simple_spread",
    *,
    render_mode: str | None = RENDER_MODE,
    **env_kwargs: Any,  # noqa: ANN401
) -> RLBenchmarkProblem:
    """
    Create a generic MPE benchmark problem from metadata only.

    Raises:
        ValueError: if env_kind is not recognized or if n_agents is incompatible with the env_kind.

    """
    if env_kind == "simple_spread":
        env_config = {
            **env_kwargs,
            "N": n_agents,
            "render_mode": render_mode,
        }
    elif env_kind == "simple_adversary":
        if n_agents < 2:
            raise ValueError("n_agents must be >= 2 for simple_adversary")
        env_config = {
            **env_kwargs,
            "N": n_agents - 1,
            "render_mode": render_mode,
        }
    else:
        raise ValueError(f"Unknown env_kind: {env_kind}")

    agents = [RLAgent(i, activation=AlwaysActive()) for i in range(n_agents)]

    return RLBenchmarkProblem(
        env_kind=env_kind,
        n_agents=n_agents,
        agents=agents,
        env_config=env_config,
    )
