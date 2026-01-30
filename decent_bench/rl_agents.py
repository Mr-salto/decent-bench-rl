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
    """ RL Agent (state container + inference + replay buffer provider + utility)"""
    def __init__(
        self,
        agent_id: int,
        *,
        action_space,
        observation_space,
        q_network = None,
        target_q_network=None,
        replay_buffer=None,
        gamma: float = 0.99,
        batch_size: int = 64,
        epsilon_schedule=None,
        activation: AgentActivationScheme,
        state_snapshot_period: int = 1,
        device: str = "cpu"
    ):
        dummy_cost = DummyCost()
        super().__init__(
            agent_id=agent_id,
            cost=dummy_cost,
            activation=activation,
            state_snapshot_period=state_snapshot_period
        )
        
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_actions = None

        self.q_network = q_network
        self.target_q_network = target_q_network
        self.device = device

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule

        self.epsilon = None
        self.global_step = 0
        self.train_step = 0

        self.episode_return = 0.0
        self.episode_length = 0
        self.last_td_loss = None

        self.recent_returns = []

        self.aux_vars["latest_obs"] = None
        self.aux_vars["done"] = False
        self.aux_vars["episode_return"] = 0.0
        self.aux_vars["episode_length"] = 0

    def act(self, obs=None, deterministic : bool = False) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            obs: observation from the environment. If None, uses aux_vars["latest_obs"].
            deterministic: if True, always select argmax action (no exploration).

        Returns:
            action: integer action compatible with Discrete action spaces.
        """
        if obs is None:
            obs = self.aux_vars["latest_obs"]

        if obs is None:
            raise RuntimeError(
                f"Agent {self.id}: act() called without an observation."
            )
        
        if self.q_network is not None:
            q_values = self.q_network(obs)

        else:
            framework, device = iop.framework_device_of_array(obs)
            q_values = iop.randn((self.n_actions,), framework, device)

        if deterministic:
            action = iop.argmax(q_values, dim=None, keepdims=False)
            return action
        
        if self.epsilon is None:
            raise RuntimeError(f"Agent {self.id}: epsilon not set before calling act().")

        if np.random.rand() < self.epsilon:
            return self.action_space.sample()

        action = iop.argmax(q_values, dim=None, keepdims=False)
        return action

    def store_transition(self, obs: Array, action: int, reward: float, next_obs : Array, done: bool, info: dict | None = None) -> int:
        """
        Store one transition in the agent's replay buffer.

        Args:
            obs: observation. If None, uses aux_vars["latest_obs"].
            action: action taken.
            reward: scalar reward.
            next_obs: next observation
            done: episode termination flag
            info: optional dictionary

        Returns:
            int: current replay buffer size after adding this transition.
        """
        if obs is None:
            obs = self.aux_vars["latest_obs"]
        if obs is None:
            raise RuntimeError(f"Agent {self.id}: store_transition() called without an observation.")

        framework, device = iop.framework_device_of_array(obs)
        obs_array = iop.to_array(obs, framework, device)
        
        if next_obs is None:
            next_obs_array = None
        else:
            framework, device = iop.framework_device_of_array(next_obs)
            next_obs_array = iop.to_array(next_obs, framework, device)

        if self.replay_buffer is None:
            self.replay_buffer = SimpleReplayBuffer(capacity=100000)

        transition = (obs_array, action, reward, next_obs_array, done, info)
        self.replay_buffer.add(transition)

        self.global_step += 1
        self.episode_return += reward
        self.episode_length += 1

        self.aux_vars["episode_return"] = self.episode_return
        self.aux_vars["episode_length"] = self.episode_length

        if done:
            self.recent_returns.append(self.episode_return)
            self.episode_return = 0.0
            self.episode_length = 0
            self.aux_vars["episode_return"] = 0.0
            self.aux_vars["episode_length"] = 0

        return self.replay_buffer.size()
