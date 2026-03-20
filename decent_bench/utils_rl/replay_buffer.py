import random
from collections import deque
from typing import Any

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.utils.array import Array

Transition = tuple[Array, int, float, Array | None, bool, dict[str, Any] | None]


class SimpleReplayBuffer:
    """
    Minimal ring buffer for testing.

    Stores transitions as tuples:
      (obs, action, reward, next_obs, done, info).
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self._buf: deque[Transition] = deque(maxlen=self.capacity)

    def add(self, transition: Transition) -> None:
        """Append one transition tuple to the replay buffer."""
        self._buf.append(transition)

    def size(self) -> int:
        """Return the number of currently stored transitions."""
        return len(self._buf)

    def clear(self) -> None:
        """Remove all transitions from the replay buffer."""
        self._buf.clear()

    def sample(self, batch_size: int) -> list[Transition]:
        """Return a list of sampled transitions (raw tuples)."""
        n = self.size()
        if batch_size >= n:
            return random.sample(list(self._buf), k=n)
        return random.sample(self._buf, k=batch_size)

    def sample_batch(self, batch_size: int) -> dict[str, Any]:  # noqa: PLR0914
        """
        Sample a minibatch and returns framework-native arrays via iop.

        Returns a dict with keys:
        - obs: stacked observations (Array) shape [B, ...]
        - actions: stacked actions (Array) shape [B]
        - rewards: stacked rewards (Array) shape [B]
        - next_obs: stacked next observations (Array) shape [B, ...]
        - dones: stacked dones (Array) shape [B]
        - infos: list of info dicts (unchanged)
        """
        transitions = self.sample(batch_size)

        obs_list: list[Array] = []
        actions_list: list[int] = []
        rewards_list: list[float] = []
        next_obs_list: list[Array | None] = []
        dones_list: list[bool] = []
        infos_list: list[dict[str, Any] | None] = []

        for obs, action, reward, next_obs, done, info in transitions:
            obs_list.append(obs)
            actions_list.append(int(action))
            rewards_list.append(float(reward))
            next_obs_list.append(next_obs)  # may be None?
            dones_list.append(bool(done))
            infos_list.append(info)

        framework, device = iop.framework_device_of_array(obs_list[0])

        obs_batch = iop.stack(obs_list, dim=0)

        next_obs_fixed = []
        for obs, next_obs in zip(obs_list, next_obs_list, strict=False):
            if next_obs is None:
                next_obs_fixed.append(iop.zeros_like(obs))
            else:
                next_obs_fixed.append(next_obs)

        next_obs_batch = iop.stack(next_obs_fixed, dim=0)

        actions_np = np.array(actions_list, dtype=np.int64)
        rewards_np = np.array(rewards_list, dtype=np.float32)
        dones_np = np.array(dones_list, dtype=np.bool_)

        actions_batch = iop.to_array(actions_np, framework, device)
        rewards_batch = iop.to_array(rewards_np, framework, device)
        dones_batch = iop.to_array(dones_np, framework, device)

        return {
            "obs": obs_batch,
            "actions": actions_batch,
            "rewards": rewards_batch,
            "next_obs": next_obs_batch,
            "dones": dones_batch,
            "infos": infos_list,
        }


class RolloutBuffer:
    """
    Simple on-policy rollout buffer.

    Designed for independent-agent A2C-style training.
    """

    def __init__(self) -> None:
        self.observations: list[Array] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.logprobs: list[float] = []
        self.values: list[float] = []

        self.advantages: Array | None = None
        self.returns: Array | None = None

    def add(  # noqa: PLR0917
        self,
        obs: Array,
        action: int,
        reward: float,
        done: bool,
        logprob: float,
        value: float,
    ) -> None:
        """Append one timestep of data."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.logprobs.append(float(logprob))
        self.values.append(float(value))

    def len(self) -> int:
        """Return the number of timesteps currently stored."""
        return len(self.rewards)

    def is_full(self, n_steps: int) -> bool:
        """Return True if collected >= n_steps."""
        return self.len() >= n_steps

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """
        Compute Generalized Advantage Estimation (GAE) advantages and returns from stored rewards/values.

        Populates self.advantages and self.returns

        Args:
            last_value: state value estimation for the last step.
            gamma: discount factor.
            gae_lambda: GAE smoothing factor.

        """
        n_steps = len(self.rewards)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)

        advantages = np.zeros(n_steps, dtype=np.float32)

        last_gae = 0.0
        next_value = float(last_value)

        for t in reversed(range(n_steps)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
            next_value = values[t]

        returns = advantages + values

        framework, device = iop.framework_device_of_array(self.observations[0])

        self.advantages = iop.to_array(advantages, framework, device)
        self.returns = iop.to_array(returns, framework, device)

    def get(self) -> dict[str, Array]:
        """
        Return rollout as iop Arrays.

        Returns a dict with keys:
        - obs: stacked observations (Array) shape [B, ...]
        - actions: stacked actions (Array) shape [B]
        - values: stacked values (Array) shape [B]
        - logprobs: stacked logprobs (Array) shape [B]
        - returns: stacked returns (Array) shape [B]
        - advantages: stacked advantages (Array) shape [B]
        - dones: stacked dones (Array) shape [B]

        Raises:
            RuntimeError: if returns and advantages are not set before computation.

        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_returns_and_advantages() before get().")
        framework, device = iop.framework_device_of_array(self.observations[0])

        actions_np = np.array(self.actions, dtype=np.int64)
        values_np = np.array(self.values, dtype=np.float32)
        logprobs_np = np.array(self.logprobs, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.bool_)

        obs_batch = iop.stack(self.observations, dim=0)
        actions_batch = iop.to_array(actions_np, framework, device)
        values_batch = iop.to_array(values_np, framework, device)
        logprobs_batch = iop.to_array(logprobs_np, framework, device)
        dones_batch = iop.to_array(dones_np, framework, device)

        return {
            "obs": obs_batch,
            "actions": actions_batch,
            "values": values_batch,
            "logprobs": logprobs_batch,
            "returns": self.returns,
            "advantages": self.advantages,
            "dones": dones_batch,
        }

    def clear(self) -> None:
        """Empty the buffer (for next rollout)."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logprobs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None
