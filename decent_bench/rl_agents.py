from decent_bench.agents import Agent
from decent_bench.costs import Cost, SumCost
from decent_bench.schemes import AgentActivationScheme

from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedFrameworks, SupportedDevices
import decent_bench.utils.interoperability as iop

from decent_bench.utils_rl.replay_buffer import SimpleReplayBuffer

import numpy as np


class DummyCost(Cost):
    """
    Minimal Cost compatible with the framework that does nothing useful
    but satisfies the Cost interface and uses the framework's Array type.

    Use as a placeholder in RLAgent.__init__ until you plug-in a real TD-loss Cost.
    """

    def __init__(self) -> None:
        self._shape = (1,)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def framework(self):
        return SupportedFrameworks.NUMPY

    @property
    def device(self):
        return SupportedDevices.CPU

    @property
    def m_smooth(self) -> float:
        return 0.0

    @property
    def m_cvx(self) -> float:
        return 0.0


    @iop.autodecorate_cost_method(Cost.function)
    def function(self, x: Array) -> float:
        return 0.0

    @iop.autodecorate_cost_method(Cost.gradient)
    def gradient(self, x: Array) -> Array:
        zero_np = np.zeros(self._shape, dtype=float)
        return iop.to_array(zero_np, self.framework, self.device)

    @iop.autodecorate_cost_method(Cost.hessian)
    def hessian(self, x: Array) -> Array:
        n = int(np.prod(self._shape))
        zero_np = np.zeros((n, n), dtype=float)
        return iop.to_array(zero_np, self.framework, self.device)

    @iop.autodecorate_cost_method(Cost.proximal)
    def proximal(self, x: Array, rho: float) -> Array:
        return iop.copy(x)

    def __add__(self, other: Cost) -> Cost:
        return SumCost([self, other])


class LinearDecreasingEpsilon:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, step: int) -> float:
        return max(0.05, 1 - step/1000)


class RLAgent(Agent):
    """
    Algorithm-agnostic RL Agent.

    Purpose:
      - container for environment specs (action_space, observation_space)
      - attachable slots for algorithm utilities (policy, value, q_network, buffers)
      - Algorithms should attach a policy/callable or networks and buffers during initialize()
      - Transient runtime values are stored in `aux_vars`
    """
    def __init__(
        self,
        agent_id: int,
        *,
        action_space=None,
        observation_space=None,
        device: str | SupportedDevices = "cpu",
        activation=None,
        state_snapshot_period: int = 1,
    ):
        cost = DummyCost()
        # cost is set to None because unused by RL algorithms.
        super().__init__(
            agent_id=agent_id,
            cost=cost,
            activation=activation,
            state_snapshot_period=state_snapshot_period,
        )
        self.action_space = action_space
        self.observation_space = observation_space

        self.device = device

        self.policy: Any = None
        self.value: Any = None

        self.q_network: Any = None
        self.target_q_network: Any = None

        self.replay_buffer: Any = None 
        self.rollout_buffer: Any = None 

        self.global_step = 0
        self.train_step = 0

        self.episode_return: float = 0.0
        self.episode_length: int = 0
        self.recent_returns: list[float] = []

        self.aux_vars["obs_at_act"] = None
        self.aux_vars["latest_obs"] = None
        self.aux_vars["last_action"] = None
        self.aux_vars["last_logprob"] = None
        self.aux_vars["last_value"] = None
        self.aux_vars["done"] = False
        self.aux_vars["episode_return"] = 0.0
        self.aux_vars["episode_length"] = 0

    def act(self, obs=None, deterministic : bool = False) -> int:
        """
        Select an action.

          1) If self.policy is set: call it (policy.get_action_and_value(obs, deterministic=False) -> (action, logprob, value))
          2) Else: use action_space.sample()

        Returns:
          - sets self.aux_vars['last_action'], ['last_logprob'], ['last_value'] as available.
          - returns the native action (int for Discrete spaces).
        """
        if obs is None:
            obs = self.aux_vars["latest_obs"]

        if obs is None:
            raise RuntimeError(f"Agent {self.id}: act() called without an observation.")

        # Save the obs used for this action so on_step_collect can retrieve the pre-step obs.
        self.aux_vars["obs_at_act"] = obs

        if self.policy is not None: 
            res = self.policy.get_action_and_value(obs, deterministic=deterministic)
            if isinstance(res, tuple) or isinstance(res, list):
                action = res[0]
                if len(res) > 1:
                    self.aux_vars["last_logprob"] = res[1]
                if len(res) > 2:
                    self.aux_vars["last_value"] = res[2]
            else:
                action = res
            
            self.aux_vars["last_action"] = action
            return action
        
        action = self.action_space.sample()
        self.aux_vars["last_action"] = action
        return action

    def store_transition(self, obs: Array, action: int, reward: float, next_obs : Array, done: bool, info: dict | None = None) -> int:
        """
        Store one transition in the agent's replay buffer if it exists.

        Args:
            obs: observation.
            action: action taken.
            reward: scalar reward.
            next_obs: next observation
            done: episode termination flag
            info: optional dictionary

        Returns:
            int: current replay buffer size after adding this transition.
        """
        if self.replay_buffer is None:
            return None

        framework, device = iop.framework_device_of_array(obs)
        obs_array = iop.to_array(obs, framework, device)
        if next_obs is not None:
            framework, device = iop.framework_device_of_array(next_obs)
            next_obs_array = iop.to_array(next_obs, framework, device)
        else:
            next_obs_array = None

        transition = (obs_array, action, reward, next_obs_array, done, info)
        self.replay_buffer.add(transition)

        self.global_step += 1
        self.episode_return += float(reward)
        self.episode_length += 1
        self.aux_vars["episode_return"] = self.episode_return
        self.aux_vars["episode_length"] = self.episode_length

        if done:
            self.recent_returns.append(self.episode_return)
            self.reset_episode_counters()

        return self.replay_buffer.size()

    def attach_policy(self, policy) -> None:
        """Attach a policy callable or policy object to this agent."""
        self.policy = policy

    def attach_value(self, value_module) -> None:
        """Attach a value (critic) module to this agent."""
        self.value = value_module

    def attach_q_network(self, qnet) -> None:
        """Attach a Q-network (for DQN-like algorithms)."""
        self.q_network = qnet

    def attach_target_q_network(self, qnet) -> None:
        """Attach a target Q-network."""
        self.target_q_network = qnet

    def attach_replay_buffer(self, buffer) -> None:
        """Attach a replay buffer instance (off-policy)."""
        self.replay_buffer = buffer

    def attach_rollout_buffer(self, buffer) -> None:
        """Attach a rollout buffer instance (on-policy)."""
        self.rollout_buffer = buffer
    
    def reset_episode_counters(self) -> None:
        """Reset per-episode counters (used when starting a new episode)."""
        self.episode_return = 0.0
        self.episode_length = 0
        self.aux_vars["episode_return"] = 0.0
        self.aux_vars["episode_length"] = 0
