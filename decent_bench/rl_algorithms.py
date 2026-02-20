import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, final, Callable
from abc import ABC, abstractmethod


import decent_bench.utils.interoperability as iop
from decent_bench.utils_rl.q_networks import QNetwork
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.utils_rl.replay_buffer import SimpleReplayBuffer
from decent_bench.utils_rl.plot_return import plot_mean_episode_return
from decent_bench.networks import Network
from decent_bench.rl_agents import RLAgent
from decent_bench.environments import PettingZooEnv



class RLAlgorithm(ABC):
    """RL algorithm - agents collaborate to solve a problem using RL."""

    @property
    @abstractmethod
    def episodes(self) -> int:
        """Number of episodes to run the algorithm for."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm."""

    @abstractmethod
    def initialize(self, agents: list[RLAgent]) -> None:
        """
        Initialize the algorithm.

        Args:
            agents: provides agents

        """

    @abstractmethod
    def step(self, agents: list[RLAgent]) -> None:
        """
        Perform one iteration of the algorithm.

        Args:
            agents: provides agents
            iteration: current iteration number

        """

    def finalize(self, agents: list[RLAgent], mean_episode_returns: list[float], env: PettingZooEnv) -> None:
        """
        Finalize the algorithm.

        Note:
            Override method as needed.
            Does not need to be implemented if no finalization is required.
            By default it is used to clean up auxiliary variables to free memory.

        Args:
            agents: provides agents

        """
        for i in agents:
            if i.aux_vars is not None:
                i.aux_vars.clear()
        plot_mean_episode_return(mean_episode_returns)
        print("Closing environment.")
        env.close()

    @final
    def run(self, agents: list[RLAgent], env: PettingZooEnv, progress_callback: Callable[[int], None] | None = None) -> None:
        """
        Run the algorithm.

        Note:
            This method first calls :meth:`initialize`, then :meth:`step` for the specified number of iterations
            and finally :meth:`finalize`.

        Warning:
            Do not override this method. Instead, override :meth:`initialize`, :meth:`step` and :meth:`finalize`
            as needed.

        Args:
            agents: provides agents
            progress_callback: optional callback to report progress after each iteration.

        """
        self.initialize(agents)
        mean_episode_returns = []
        
        for episode in range(self.episodes):
            print(f"\n===== EPISODE {episode} =====")
            action_dict = {}
            obs = env.reset()
            N_STEPS = 30
            for step in range(N_STEPS): # Pass N_steps per episode as arg somewhere
                for agent in agents:
                    obs[agent] = agent.aux_vars["latest_obs"]
                    action = agent.act(obs=obs[agent], deterministic=False)
                    action_dict[agent] = action

                results = env.step(action_dict)

                for agent, (next_obs, reward, done, info) in results.items():
                    buffer_size = agent.store_transition(
                        obs[agent],
                        action_dict[agent],
                        reward,
                        next_obs,
                        done,
                    )

                if any(entry[2] for entry in results.values()):
                    per_agent_returns = []
                    for agent in agents:
                        per_agent_returns.append(agent.recent_returns[-1])
                    mean_episode_returns.append(np.mean(per_agent_returns))
                    break

                self.step(agents)

            if progress_callback is not None:
                progress_callback(k)

        self.finalize(agents, mean_episode_returns, env)


@dataclass(eq=False)
class IndependentDQN(RLAlgorithm):
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
    episodes: int = 100
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

    def initialize(self, agents: list[RLAgent]) -> None:
        for agent in agents:
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

    def step(self, agents: list[RLAgent]) -> None:
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

        for agent in agents:
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
