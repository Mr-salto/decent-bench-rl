from __future__ import annotations
from typing import Sequence, Dict, Any, Callable
import numpy as np

class PettingZooEnv:
    """
    Minimal wrapper connecting decent_bench's Agent objects
    with a generic PettingZoo Parallel environment.
    
    The wrapper:
      - creates the environment
      - maps Agent[i] -> env.agents[i]
      - stores action/obs spaces
      - provides reset(), step(), render(), close()
    """

    def __init__(self, agents: list[Agent], env_factory: Callable[..., Any], **env_kwargs):
        """
        Args:
            agents: a sequence of Agent instances.
            env_factory: a callable returning a PettingZoo ParallelEnv
                         (e.g., simple_spread_v3.parallel_env).
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

        self.agent_to_env_name = {agent: name for agent, name in zip(self.agents, self.env_agent_names)}
        self.env_name_to_agent = {name: agent for agent, name in self.agent_to_env_name.items()}

        try:
            self.observation_spaces = {name: self.env.observation_space(name) for name in self.env_agent_names}
            self.action_spaces = {name: self.env.action_space(name) for name in self.env_agent_names}
        except Exception:
            self.observation_spaces = {}
            self.action_spaces = {}

        self.last_obs: Dict[str, Any] = {}
        self.last_rewards: Dict[str, float] = {}
        self.last_dones: Dict[str, bool] = {}
        self.last_infos: Dict[str, dict] = {}

    def reset(self, **kwargs):
        """
        Reset the underlying PettingZoo environment.

        Returns:
            A dict mapping Agent -> initial observation.
        """
        # Parallel API returns a tuple of dicts: tuple[dict[AgentID, ObsType], dict[AgentID, dict]]
        obs, info = self.env.reset(**kwargs)  
        self.last_obs = {}
        self.last_rewards = {name: 0.0 for name in self.env_agent_names}
        self.last_dones = {name: False for name in self.env_agent_names}
        self.last_infos = {name: {} for name in self.env_agent_names}

        agent_obs = {}
        for env_name, observation in obs.items():
            agent = self.env_name_to_agent[env_name]

            self.last_obs[env_name] = observation
            agent.aux_vars["latest_obs"] = observation

            agent_obs[agent] = observation

        return agent_obs
    
    def step(self, action_dict):
        """
        Take a step in the underlying PettingZoo environment.

        Args:
            action_dict: dict mapping Agent -> action

        Returns:
            dict mapping Agent -> (obs, reward, done, info)
        """

        env_actions = {
            self.agent_to_env_name[agent]: action
            for agent, action in action_dict.items()
        }

        obs_dict, reward_dict, done_dict, _ , info_dict = self.env.step(env_actions)

        self.last_obs = obs_dict.copy()
        self.last_rewards = reward_dict.copy()
        self.last_dones = done_dict.copy()
        self.last_infos = info_dict.copy()

        results = {}

        for env_name in self.env_agent_names:
            agent = self.env_name_to_agent[env_name]

            obs = obs_dict.get(env_name)
            rew = reward_dict.get(env_name, 0.0)
            done = done_dict.get(env_name, False)
            info = info_dict.get(env_name, {})

            agent.aux_vars["latest_obs"] = obs

            results[agent] = (obs, rew, done, info)

        return results
    
    def render(self):
        """
        Render the underlying PettingZoo environment.
        """
        return self.env.render()

    def close(self):
        """
        Close the environment and free resources.
        """
        if hasattr(self.env, "close"):
            self.env.close()
