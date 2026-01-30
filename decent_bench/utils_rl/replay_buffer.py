from collections import deque
import random
from decent_bench.utils.array import Array
from typing import Tuple, List
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
