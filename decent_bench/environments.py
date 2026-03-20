from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.rl_agents import RLAgent
from decent_bench.utils.array import Array

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class PettingZooEnv:
    """
    Minimal wrapper connecting decent_bench's Agent objects with a generic PettingZoo Parallel environment.

    The wrapper:
      - creates the environment
      - maps Agent[i] -> env.agents[i]
      - stores action/obs spaces
      - provides reset(), step(), render(), close()

    """

    def __init__(
        self,
        agents: list[RLAgent],
        env_factory: Callable[..., Any],
        **env_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """
        Initialize the PettingZoo wrapper and validate agent mapping.

        Args:
            agents: a sequence of Agent instances.
            env_factory: a callable returning a PettingZoo ParallelEnv
                         (e.g., simple_spread_v3.parallel_env).
            **env_kwargs: keyword arguments forwarded to env_factory.

        Raises:
            AttributeError: if the environment exposes neither agents nor possible_agents.
            ValueError: if the number of RL agents does not match environment agents.

        """
        self.agents = list(agents)

        self.env = env_factory(**env_kwargs)

        if hasattr(self.env, "agents"):
            self.env_agent_names = list(self.env.agents)
        elif hasattr(self.env, "possible_agents"):
            self.env_agent_names = list(self.env.possible_agents)
        else:
            raise AttributeError(
                "Environment object does not expose 'agents' or 'possible_agents'. "
                "Pass a Parallel API env (env.agents) or an appropriate PettingZoo env."
            )

        if len(self.agents) != len(self.env_agent_names):
            raise ValueError(
                f"Number of provided Agents ({len(self.agents)}) does not match "
                f"environment agents ({len(self.env_agent_names)}: {self.env_agent_names})."
            )

        self.agent_to_env_name = dict(zip(self.agents, self.env_agent_names, strict=False))
        self.env_name_to_agent = {name: agent for agent, name in self.agent_to_env_name.items()}

        try:
            self.observation_spaces = {name: self.env.observation_space(name) for name in self.env_agent_names}
            self.action_spaces = {name: self.env.action_space(name) for name in self.env_agent_names}
        except Exception:
            self.observation_spaces = {}
            self.action_spaces = {}

        self.last_obs: dict[str, Any] = {}
        self.last_rewards: dict[str, float] = {}
        self.last_dones: dict[str, bool] = {}
        self.last_infos: dict[str, dict[str, Any]] = {}

    def reset(self, **kwargs: Any) -> dict[RLAgent, Array]:  # noqa: ANN401
        """
        Reset the underlying PettingZoo environment.

        Returns:
            A dict mapping Agent -> initial observation.

        """
        # Parallel API returns a tuple of dicts: tuple[dict[AgentID, ObsType], dict[AgentID, dict]]
        obs, _info = self.env.reset(**kwargs)
        self.last_obs = {}
        self.last_rewards = dict.fromkeys(self.env_agent_names, 0.0)
        self.last_dones = dict.fromkeys(self.env_agent_names, False)
        self.last_infos = {name: {} for name in self.env_agent_names}

        agent_obs: dict[RLAgent, Array] = {}
        for env_name, observation in obs.items():
            agent = self.env_name_to_agent[env_name]

            self.last_obs[env_name] = observation

            framework, device = iop.framework_device_of_array(observation)
            obs_array = iop.to_array(observation, framework, device)

            agent.latest_obs = obs_array
            agent_obs[agent] = obs_array

        return agent_obs

    def step(  # noqa: PLR0914
        self,
        action_dict: Mapping[RLAgent, int | Array],
    ) -> dict[RLAgent, tuple[Array | None, float, bool, dict[str, Any]]]:
        """
        Take a step in the underlying PettingZoo environment.

        Args:
            action_dict: dict mapping Agent -> action

        Returns:
            dict mapping Agent -> (obs, reward, done, info)

        """
        env_actions = {
            self.agent_to_env_name[agent]: self._array_to_int(action) for agent, action in action_dict.items()
        }
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(env_actions)
        self.last_obs = obs_dict.copy()
        self.last_rewards = reward_dict.copy()
        self.last_infos = info_dict.copy()

        results: dict[RLAgent, tuple[Array | None, float, bool, dict[str, Any]]] = {}

        for env_name in self.env_agent_names:
            agent = self.env_name_to_agent[env_name]

            obs = obs_dict.get(env_name)
            rew = reward_dict.get(env_name, 0.0)
            terminated = terminated_dict.get(env_name, False)
            truncated = truncated_dict.get(env_name, False)
            done = terminated or truncated
            info = info_dict.get(env_name, {})

            if obs is not None:
                framework, device = iop.framework_device_of_array(obs)
                obs_array = iop.to_array(obs, framework, device)
                agent.latest_obs = obs_array
            else:
                obs_array = None

            results[agent] = (obs_array, rew, done, info)

        return results

    def render(self) -> Any:  # noqa: ANN401
        """Render the underlying PettingZoo environment."""
        return self.env.render()

    def close(self) -> None:
        """Close the environment and free resources."""
        if hasattr(self.env, "close"):
            self.env.close()

    def _array_to_int(self, array: int | Array) -> int:
        val = array.value if isinstance(array, Array) else array
        if isinstance(val, np.ndarray | np.generic):
            return int(val.item())

        if TORCH_AVAILABLE and isinstance(val, torch.Tensor):
            return int(val.item())
        return int(val)
