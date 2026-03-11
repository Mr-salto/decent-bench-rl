from collections import deque
import random
from decent_bench.utils.array import Array
from typing import Tuple, List, Any, Dict, Optional
import numpy as np
import decent_bench.utils.interoperability as iop


class SimpleReplayBuffer:
    """
    Minimal ring buffer for testing.
    Stores transitions as tuples:
      (obs, action, reward, next_obs, done, info)
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self._buf = deque(maxlen=self.capacity)

    def add(self, transition: Tuple[Array, int, float, Array, bool, dict | None]) -> None:
        self._buf.append(transition)

    def size(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def sample(self, batch_size: int) -> list[Tuple[Array, int, float, Array | None, bool, dict | None]]:
        """Return a list of sampled transitions (raw tuples)."""
        n = self.size()
        if batch_size >= n:
            return random.sample(list(self._buf), k=n)
        return random.sample(self._buf, k=batch_size)

    def sample_batch(self, B: int) -> dict[str, any]:
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
        transitions = self.sample(B)

        obs_list: List[Array] = []
        actions_list: List[int] = []
        rewards_list: List[float] = []
        next_obs_list: List[Array] = []
        dones_list: List[bool] = []
        infos_list: List[Optional[dict]] = []

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
        for obs, next_obs in zip(obs_list, next_obs_list):
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
    Simple on-policy rollout buffer
    Designed for independent-agent A2C-style training.
    """
    def __init__(
        self,
        gae_lambda: float = 1,
        gamma: float = 1,
        ) -> None:
        self.observations: List[Array] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.logprobs: List[float] = []
        self.values: List[float] = []

        self.advantages: Optional[Array] = None
        self.returns: Optional[Array] = None

        self.gae_lambda = gae_lambda
        self.gamma = gamma

    def add(self,
        obs: Array,
        action: int,
        reward: float,
        done: bool,
        logprob: float,
        value: float,
        ) -> None:
        """Append one timestep of data"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.logprobs.append(float(logprob))
        self.values.append(float(value))

    def len(self) -> int:
        """Number of timesteps stored"""
        return len(self.rewards)

    def is_full(self, n_steps: int) -> bool:
        """Return True if collected >= n_steps."""
        return self.len() >= n_steps

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """
        Compute Generalized Advantage Estimation (GAE) advantages and returns from stored rewards/values.
        Populates self.advantages and self.returns
        Args: 
            last_value: state value estimation for the last step
        """
        T = len(self.rewards)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        
        last_gae = 0.0
        next_value = float(last_value)

        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
            next_value = values[t]

        returns = advantages + values

        framework, device = iop.framework_device_of_array(self.observations[0])

        self.advantages = iop.to_array(advantages, framework, device)
        self.returns = iop.to_array(returns, framework, device)

    def get(self) -> Dict[str, Array]:
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
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError(
                "Call compute_returns_and_advantages() before get()."
            )        
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
