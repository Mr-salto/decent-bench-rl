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

StepResult = tuple[Array | None, float, bool, dict[str, Any]]


class MPE:
    """
    Minimal wrapper connecting decent_bench's Agent objects with a generic MPE Parallel environment.

    The wrapper:
      - creates the environment
      - maps Agent[i] -> env.agents[i]
      - stores action/obs spaces
      - provides reset(), step(), render(), close()

    """

    def __init__(
        self,
        env_factory: Callable[..., Any],
        agents: list[RLAgent] | None = None,
        **env_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """
        Initialize the MPE wrapper and validate agent mapping.

        Args:
            agents: optional list of Agent instances. If omitted, call attach_agents(...)
                before reset()/step().
            env_factory: a callable returning an MPE ParallelEnv
                         (e.g., simple_spread_v3.parallel_env).
            **env_kwargs: keyword arguments forwarded to env_factory.

        Raises:
            AttributeError: if the environment exposes neither agents nor possible_agents.
            RuntimeError: if action space / observation space initialization fails.

        """
        self.agents: list[RLAgent] = []

        self.env = env_factory(**env_kwargs)

        # Some Parallel envs expose `agents` but keep it empty before reset().
        if hasattr(self.env, "agents") and self.env.agents:
            self.env_agent_names = list(self.env.agents)
        elif hasattr(self.env, "possible_agents"):
            self.env_agent_names = list(self.env.possible_agents)
        elif hasattr(self.env, "agents"):
            self.env_agent_names = list(self.env.agents)
        else:
            raise AttributeError(
                "Environment object does not expose 'agents' or 'possible_agents'. "
                "Pass a Parallel API env (env.agents) or an appropriate MPE env."
            )

        self.agent_to_env_name: dict[RLAgent, str] = {}
        self.env_name_to_agent: dict[str, RLAgent] = {}

        try:
            self.observation_spaces = {name: self.env.observation_space(name) for name in self.env_agent_names}
            self.action_spaces = {name: self.env.action_space(name) for name in self.env_agent_names}
        except Exception as e:
            raise RuntimeError("Failed to initialize observation_spaces/action_spaces from MPE") from e

        self.last_obs: dict[str, Any] = {}
        self.last_rewards: dict[str, float] = {}
        self.last_dones: dict[str, bool] = {}
        self.last_infos: dict[str, dict[str, Any]] = {}
        self.last_global_state: Array | None = None

        if agents is not None:
            self.attach_agents(agents)

    def attach_agents(self, agents: list[RLAgent]) -> None:
        """
        Attach RL agents to environment agent names.

        Raises:
            ValueError: if the number of RL agents does not match environment agents.

        """
        self.agents = list(agents)

        if len(self.agents) != len(self.env_agent_names):
            raise ValueError(
                f"Number of provided Agents ({len(self.agents)}) does not match "
                f"environment agents ({len(self.env_agent_names)}: {self.env_agent_names})."
            )

        self.agent_to_env_name = dict(zip(self.agents, self.env_agent_names, strict=False))
        self.env_name_to_agent = {name: agent for agent, name in self.agent_to_env_name.items()}

    def _require_attached_agents(self) -> None:
        if not self.agent_to_env_name:
            raise RuntimeError(
                "MPE has no attached agents. "
                "Instantiate with agents=... or call attach_agents(...) before reset()/step()."
            )

    def reset(self, **kwargs: Any) -> dict[RLAgent, Array]:  # noqa: ANN401
        """
        Reset the underlying MPE environment.

        Returns:
            A dict mapping Agent -> initial observation.

        """
        self._require_attached_agents()
        # Parallel API returns a tuple of dicts: tuple[dict[AgentID, ObsType], dict[AgentID, dict]]
        obs, _info = self.env.reset(**kwargs)
        self.last_obs = {}
        self.last_rewards = dict.fromkeys(self.env_agent_names, 0.0)
        self.last_dones = dict.fromkeys(self.env_agent_names, False)
        self.last_infos = {name: {} for name in self.env_agent_names}
        for agent in self.agents:
            agent.done = False

        agent_obs: dict[RLAgent, Array] = {}
        for env_name, observation in obs.items():
            agent = self.env_name_to_agent[env_name]

            self.last_obs[env_name] = observation

            framework, device = iop.framework_device_of_array(observation)
            obs_array = iop.to_array(observation, framework, device)

            agent.latest_obs = obs_array
            agent_obs[agent] = obs_array

        self.last_global_state = self.get_global_state()

        return agent_obs

    def step(  # noqa: PLR0914
        self,
        action_dict: Mapping[RLAgent, int | Array],
    ) -> tuple[dict[RLAgent, StepResult], bool, float | None]:
        """
        Take a step in the underlying MPE environment.

        Args:
            action_dict: dict mapping Agent -> action

        Returns:
            Tuple containing step results, episode completion status, and mean episode return.

        """
        self._require_attached_agents()
        state_before = self.last_global_state
        env_actions = {
            self.agent_to_env_name[agent]: self._array_to_int(action) for agent, action in action_dict.items()
        }
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(env_actions)
        self.last_obs = obs_dict.copy()
        self.last_rewards = reward_dict.copy()
        self.last_infos = info_dict.copy()
        next_state = self.get_global_state()
        self.last_global_state = next_state

        results: dict[RLAgent, StepResult] = {}
        done_by_env_name: dict[str, bool] = {}

        for env_name in self.env_agent_names:
            agent = self.env_name_to_agent[env_name]

            obs = obs_dict.get(env_name)
            rew = reward_dict.get(env_name, 0.0)
            terminated = terminated_dict.get(env_name, False)
            truncated = truncated_dict.get(env_name, False)
            done = terminated or truncated
            info = dict(info_dict.get(env_name, {}))

            if state_before is not None:
                info["global_state"] = state_before
            if next_state is not None:
                info["next_global_state"] = next_state
            done_by_env_name[env_name] = done

            if obs is not None:
                framework, device = iop.framework_device_of_array(obs)
                obs_array = iop.to_array(obs, framework, device)
                agent.latest_obs = obs_array
            else:
                obs_array = None

            results[agent] = (obs_array, rew, done, info)
            self.last_infos[env_name] = info

        self.last_dones = done_by_env_name.copy()

        episode_done = self._apply_step_accounting(results)
        mean_episode_return = self._finalize_episode_stats() if episode_done else None

        return results, episode_done, mean_episode_return

    def get_global_state(self) -> Array | None:
        """
        Retrieve a global state representation from the underlying environment.

        Returns:
            iop Array for the global state, or None if unavailable.

        """
        state = self.env.state()

        if state is None:
            return None

        framework, device = iop.framework_device_of_array(state)
        return iop.to_array(state, framework, device)

    def render(self) -> Any:  # noqa: ANN401
        """Render the underlying MPE environment."""
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

    def _apply_step_accounting(self, results: dict[RLAgent, StepResult]) -> bool:
        """Update generic per-step counters on all agents and return whether an episode ended."""
        episode_done = False
        for agent, (_next_obs, reward, done, _info) in results.items():
            agent.global_step += 1
            agent.episode_return += float(reward)
            agent.episode_length += 1
            agent.done = bool(done)
            episode_done = episode_done or bool(done)
        return episode_done

    def _finalize_episode_stats(self) -> float:
        """Finalize all agent episode stats exactly once and return their mean return."""
        per_agent_returns: list[float] = []
        for agent in self.agents:
            if agent.episode_length > 0:
                per_agent_returns.append(float(agent.finalize_episode()))
            elif agent.recent_returns:
                per_agent_returns.append(float(agent.recent_returns[-1]))
            else:
                per_agent_returns.append(float(agent.finalize_episode()))
            agent.done = False

        return float(np.mean(per_agent_returns))
