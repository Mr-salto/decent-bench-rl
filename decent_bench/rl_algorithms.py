import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

import decent_bench.utils.interoperability as iop
from decent_bench.utils_rl.q_networks import QNetwork
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.utils_rl.replay_buffer import SimpleReplayBuffer
from decent_bench.networks import Network


@dataclass(eq=False)
class IndependentDQN(Algorithm):
    """
    Independent DQN (IDQN) algorithm — each agent trains its own Q-network using
    its local replay buffer. Implements initialize() and step().

    Hyperparameters:
      - gamma: discount factor
      - lr: optimizer learning rate
      - batch_size: minibatch size
      - replay_start_size: minimal buffer size to start training
      - target_update_freq: number of training updates between target syncs
      - train_freq: perform a training update every `train_freq` environment steps (1 = every step)
      - grad_updates_per_step: number of gradient steps per training trigger (often 1)
      - device: "cpu" or "cuda"
    """

    iterations: int = 100
    name: str = "IDQN"

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 32
    replay_start_size: int = 500
    target_update_freq: int = 100
    train_freq: int = 1
    grad_updates_per_step: int = 1
    name: str = "IndependentDQN"
    device: str = "cpu"

    def initialize(self, network: Network) -> None:
        for agent in network.agents():
            torch_device = torch.device(self.device)

            if agent.q_network is None:
                obs_space = agent.observation_space
                if obs_space is None:
                    raise RuntimeError(f"Agent {agent.id} has no observation_space set during DQN.initialize()")
                obs_dim = int(np.prod(obs_space.shape))
                agent.q_network = QNetwork(
                    obs_dim=obs_dim, n_actions=agent.n_actions, hidden_sizes=(64, 64), device=self.device
                )

            if agent.target_q_network is None:
                obs_space = agent.observation_space
                obs_dim = int(np.prod(obs_space.shape))
                agent.target_q_network = QNetwork(
                    obs_dim=obs_dim, n_actions=agent.n_actions, hidden_sizes=(64, 64), device=self.device
                )
                agent.target_q_network.copy_from(agent.q_network)

            if "optimizer" not in agent.aux_vars or agent.aux_vars["optimizer"] is None:
                opt = torch.optim.Adam(agent.q_network.parameters(), lr=self.lr)
                agent.aux_vars["optimizer"] = opt

            # Initialize training step counters
            agent.train_step = agent.train_step
            agent.aux_vars["dqn_train_steps"] = 0

            if agent.replay_buffer is None:
                agent.replay_buffer = SimpleReplayBuffer(capacity=100000)

    def step(self, network: Network, iteration: int) -> None:
        """
        One iteration: for each active agent, possibly perform training updates if buffer is ready.
         - update epsilon from schedule (if present) using agent.global_step
         - if replay buffer is large enough and agent.global_step % train_freq == 0:
             - sample batch
             - convert batch fields to torch tensors on the agent device
             - compute q_pred (gather by actions)
             - compute q_target using agent.target_q_network (no grad)
             - compute loss (MSE) and backprop / optimizer step
             - repeat grad_updates_per_step times
             - periodically sync target network
        """

        for agent in network.active_agents(iteration):
            if agent.epsilon_schedule is not None and callable(agent.epsilon_schedule):
                agent.epsilon = float(agent.epsilon_schedule(agent.global_step))

            replay_buffer = agent.replay_buffer
            replay_size = replay_buffer.size()
            global_step = agent.global_step

            if replay_size < self.replay_start_size or global_step % self.train_freq != 0:
                continue

            optimizer = agent.aux_vars["optimizer"]

            for _ in range(self.grad_updates_per_step):
                batch = replay_buffer.sample_batch(self.batch_size)

                obs_t = iop.to_torch(batch["obs"], agent.q_network.torch_device)
                next_obs_t = iop.to_torch(batch["next_obs"], agent.q_network.torch_device)
                actions_t = iop.to_torch(batch["actions"], agent.q_network.torch_device)
                rewards_t = iop.to_torch(batch["rewards"], agent.q_network.torch_device)
                dones_t = iop.to_torch(batch["dones"], agent.q_network.torch_device)
                dones_t = dones_t.float()

                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                if next_obs_t is not None and next_obs_t.dim() == 1:
                    next_obs_t = next_obs_t.unsqueeze(0)

                agent.q_network.torch_module.train()
                q_values = agent.q_network.torch_module(obs_t)  # [B, #actions]
                q_pred = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(
                    1
                )  # q_pred[b] == q_values[b, actions_t[b]] for each sample b

                with torch.no_grad():
                    agent.target_q_network.torch_module.eval()
                    next_q_values = agent.target_q_network.torch_module(next_obs_t)  # [B, A]
                    next_q_max, _ = next_q_values.max(dim=1)  # [B]
                    td_target = rewards_t + self.gamma * (1.0 - dones_t) * next_q_max

                loss = F.mse_loss(q_pred, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                agent.train_step += 1
                agent.aux_vars["dqn_train_steps"] += 1

                if (agent.aux_vars["dqn_train_steps"] % self.target_update_freq) == 0:
                    agent.target_q_network.copy_from(agent.q_network)
