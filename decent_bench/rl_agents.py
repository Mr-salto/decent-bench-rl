from dataclasses import dataclass
from typing import Any

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent  # from decent_bench import agent
from decent_bench.costs import Cost, SumCost
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks
from decent_bench.utils_rl.q_networks import BaseNetwork, QNetwork
from decent_bench.utils_rl.replay_buffer import RolloutBuffer, SimpleReplayBuffer


class _RLCostAdapter(Cost):
    """Internal adapter used only to satisfy Agent's constructor in RL mode."""

    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def framework(self) -> SupportedFrameworks:
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:
        return SupportedDevices.CPU

    @property
    def m_smooth(self) -> float:
        return 0.0

    @property
    def m_cvx(self) -> float:
        return 0.0

    def function(self, x: Array, **kwargs: Any) -> float:  # noqa: ARG002, ANN401
        raise RuntimeError("RLAgent does not use optimization costs.")

    def gradient(self, x: Array, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        raise RuntimeError("RLAgent does not use optimization costs.")

    def hessian(self, x: Array, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        raise RuntimeError("RLAgent does not use optimization costs.")

    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        raise RuntimeError("RLAgent does not use optimization costs.")

    def __add__(self, other: Cost) -> Cost:
        return SumCost([self, other])


class LinearDecreasingEpsilon:
    """Linear decay schedule with a fixed minimum epsilon of 0.05."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, step: int) -> float:
        """Return epsilon value for the provided global step."""
        return max(0.05, 1 - step / 1000)


# class RLAgent(agents.Agent):
class RLAgent(Agent):
    """
    Algorithm-agnostic RL Agent.

    Purpose:
      - container for environment specs (action_space, observation_space)
      - attachable slots for algorithm utilities (policy, value, q_network, buffers)
      - Algorithms should attach a policy/callable or networks and buffers during initialize()
      - Transient runtime values are stored on explicit RLAgent attributes
    """

    def __init__(
        self,
        agent_id: int,
        *,
        action_space: Any = None,  # noqa: ANN401
        observation_space: Any = None,  # noqa: ANN401
        device: str | SupportedDevices = "cpu",
        activation: Any = None,  # noqa: ANN401
        state_snapshot_period: int = 1,
    ):
        cost = _RLCostAdapter()
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

        self.q_network: Any = None
        self.target_q_network: Any = None

        self.replay_buffer: Any = None
        self.rollout_buffer: Any = None

        self.global_step = 0
        self.train_step = 0
        self.dqn_train_steps: int = 0

        self.episode_return: float = 0.0
        self.episode_length: int = 0
        self.recent_returns: list[float] = []
        self.episode_return_history: list[float] = []

        # RL-specific runtime state should not be stored in Agent.aux_vars because
        # Agent types aux_vars values as Array.
        self.obs_at_act: Array | None = None
        self.latest_obs: Array | None = None
        self.last_action: int | None = None
        self.last_logprob: Any = None
        self.last_value: Any = None
        self.done: bool = False
        self.optimizer: Any = None

    def act(self, obs: Array | None = None, deterministic: bool = False) -> int:
        """
        Select an action.

        1) If self.policy is set, call
              policy.get_action_and_value(obs, deterministic=False)
              -> (action, logprob, value).
        2) Else: use action_space.sample()

        Returns:
            - stores the selected action and optional policy outputs on the agent.
            - returns the native action (int for Discrete spaces).

        Raises:
            RuntimeError: if act() is called with no observation available.

        """
        if obs is None:
            obs = self.latest_obs

        if obs is None:
            raise RuntimeError(f"Agent {self.id}: act() called without an observation.")

        # Save the obs used for this action so on_step_collect can retrieve the pre-step obs.
        self.obs_at_act = obs

        if self.policy is not None:
            res = self.policy.get_action_and_value(obs, deterministic=deterministic)
            if isinstance(res, (tuple, list)):
                action = int(res[0])
                if len(res) > 1:
                    self.last_logprob = res[1]
                if len(res) > 2:
                    self.last_value = res[2]
            else:
                action = int(res)

            self.last_action = action
            return action

        action = int(self.action_space.sample())
        self.last_action = action
        return action

    def store_transition(  # noqa: PLR0917
        self,
        obs: Array,
        action: int,
        reward: float,
        next_obs: Array | None,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> int | None:
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

        return int(self.replay_buffer.size())

    def attach_policy(self, policy: BaseNetwork) -> None:
        """Attach a policy callable or policy object to this agent."""
        self.policy = policy

    def attach_q_network(self, qnet: QNetwork) -> None:
        """Attach a Q-network (for DQN-like algorithms)."""
        self.q_network = qnet

    def attach_target_q_network(self, qnet: QNetwork) -> None:
        """Attach a target Q-network."""
        self.target_q_network = qnet

    def attach_replay_buffer(self, buffer: SimpleReplayBuffer) -> None:
        """Attach a replay buffer instance (off-policy)."""
        self.replay_buffer = buffer

    def attach_rollout_buffer(self, buffer: RolloutBuffer) -> None:
        """Attach a rollout buffer instance (on-policy)."""
        self.rollout_buffer = buffer

    def reset_episode_counters(self) -> None:
        """Reset per-episode counters (used when starting a new episode)."""
        self.episode_return = 0.0
        self.episode_length = 0

    def finalize_episode(self) -> float:
        """
        Persist per-episode statistics and reset episode counters.

        Returns:
            The recorded episode return before counters are reset.

        """
        episode_return = float(self.episode_return)
        self.recent_returns.append(episode_return)
        self.episode_return_history.append(episode_return)
        self.reset_episode_counters()
        return episode_return


@dataclass(frozen=True, eq=False)
class RLAgentMetricsView:
    """Immutable view of RL agent that exposes useful properties for calculating metrics."""

    episode_return_history: list[float]
    global_step: int
    train_step: int

    @staticmethod
    def from_agent(agent: RLAgent) -> "RLAgentMetricsView":
        """Create from agent."""
        return RLAgentMetricsView(
            episode_return_history=agent.episode_return_history,
            global_step=agent.global_step,
            train_step=agent.train_step,
        )
