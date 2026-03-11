import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from dataclasses import dataclass
from typing import Optional, List, final, Callable, Dict
from abc import ABC, abstractmethod


import decent_bench.utils.interoperability as iop
from decent_bench.utils_rl.q_networks import DQNPolicy
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.utils_rl.replay_buffer import SimpleReplayBuffer
from decent_bench.utils_rl.plot_return import plot_mean_episode_return
from decent_bench.networks import Network
from decent_bench.rl_agents import RLAgent, LinearDecreasingEpsilon
from decent_bench.environments import PettingZooEnv



class RLAlgorithm(ABC):
    """RL algorithm - agents collaborate to solve a problem using RL."""

    @property
    @abstractmethod
    def episodes(self) -> int:
        """Number of episodes to run the algorithm for."""
    
    @property
    @abstractmethod
    def episode_length(self) -> int:
        """Number of environment steps per episode to run the algorithm for."""

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

    def on_step_collect(self, agents: list[RLAgent], results: Dict[RLAgent, tuple]) -> None:
        """
        Called immediately after env.step(...) and after storing transitions.
        Used for on-policy collectors, logging, or any per-step bookkeeping.
        `results` is mapping agent -> (next_obs, reward, done, info)
        """
        return None

    def on_step_update(self, agents: list[RLAgent]) -> None:
        """
        Called once per environment step after on_step_collect.
        Used for off-policy updates.
        """
        return None

    def on_episode_end(self, agents: list[RLAgent]) -> None:
        """
        Called once when an episode ends (any agent done). Useful for on-policy algorithms
        that need to compute advantages and perform batch updates.
        """
        return None

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
    def run(self, agents: list[RLAgent], env: PettingZooEnv) -> None:
        """
        Run the algorithm.

        Note:
            This method first calls :meth:`initialize`, then :meth:`step` for the specified number of episodes
            and finally :meth:`finalize`. Relies on store_transition t increment agent.globel_step.

        Warning:
            Do not override this method. Instead, override :meth:`initialize`, :meth:`step` and :meth:`finalize`
            as needed.

        Args:
            agents: provides agents
            env: environment to run the algorithm on.
        """
        self.initialize(agents)
        mean_episode_returns = []
        
        for episode in range(self.episodes):
            print(f"\n===== EPISODE {episode} =====")
            print("epsilon = ", agents[0].policy.epsilon)
            action_dict = {}
            obs = env.reset()
            episode_done = False

            for step in range(self.episode_length):
                for agent in agents:
                    obs[agent] = agent.aux_vars["latest_obs"]
                    action = agent.act(obs=obs[agent], deterministic=False)
                    action_dict[agent] = action
                    agent.aux_vars["last_action"] = action

                results = env.step(action_dict)
            
                self.on_step_collect(agents, results)

                self.on_step_update(agents)

                if any(entry[2] for entry in results.values()):
                    per_agent_returns = []
                    for agent in agents:
                        per_agent_returns.append(agent.recent_returns[-1])
                    mean_episode_returns.append(np.mean(per_agent_returns))
                    self.on_episode_end(agents)
                    episode_done = True
                    break
                
            # Append returns for episodes that end due to N_STEPS reaching max and not due to agents being done
            if not episode_done:
                per_agent_returns = []
                for agent in agents:
                    per_agent_returns.append(agent.recent_returns[-1])
                mean_episode_returns.append(np.mean(per_agent_returns))
                self.on_episode_end(agents)

        self.finalize(agents, mean_episode_returns, env)
        return mean_episode_returns


@dataclass(eq=False)
class IndependentDQN(RLAlgorithm):
    """
    Independent DQN (IDQN) algorithm — each agent trains its own Q-network using
    its local replay buffer.

    Each agent gets a DQNPolicy attached as agent.policy, which encapsulates the
    online and target Q-networks together with epsilon-greedy action selection.
    RLAgent.act() calls policy.get_action_and_value() automatically.

    Hyperparameters:
      - gamma: discount factor
      - lr: optimizer learning rate
      - batch_size: minibatch size
      - replay_start_size: minimal buffer size before training begins
      - target_update_freq: number of gradient updates between target hard-syncs
      - train_freq: perform a gradient update every `train_freq` env steps
      - grad_updates_per_step: gradient steps per training trigger
      - device: 'cpu' or 'cuda'
    """
    episodes: int = 100
    episode_length: int = 30
    name: str = "IDQN"

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 32
    replay_start_size: int = 500
    target_update_freq: int = 100
    train_freq: int = 1
    grad_updates_per_step: int = 1
    device: str = "cpu"

    def initialize(self, agents: list[RLAgent]) -> None:
        for agent in agents:
            obs_space = agent.observation_space
            if obs_space is None:
                raise RuntimeError(
                    f"Agent {agent.id} has no observation_space set during DQN.initialize()"
                )
            obs_dim = int(np.prod(obs_space.shape))
            n_actions = agent.action_space.n
            agent.n_actions = n_actions

            policy = DQNPolicy(
                obs_dim=obs_dim,
                n_actions=n_actions,
                hidden_sizes=(64, 64),
                epsilon=1.0,
                epsilon_schedule=LinearDecreasingEpsilon(1.0),
                device=self.device,
            )
            agent.attach_policy(policy)

            opt = torch.optim.Adam(policy.parameters(), lr=self.lr)
            agent.aux_vars["optimizer"] = opt
            agent.aux_vars["dqn_train_steps"] = 0

            if agent.replay_buffer is None:
                agent.replay_buffer = SimpleReplayBuffer(capacity=100000)

    def on_step_collect(self, agents: list[RLAgent], results: Dict[RLAgent, tuple]) -> None:
        """IDQN collects transitions by calling agent.store_transition()."""
        for agent, (next_obs, reward, done, info) in results.items():
            prev_obs = agent.aux_vars["latest_obs"]
            action = agent.aux_vars["last_action"]
            agent.store_transition(prev_obs, action, reward, next_obs, done, info)

    def on_step_update(self, agents: list[RLAgent]) -> None:
        """
        For each agent: decay epsilon, then (if buffer is ready) run gradient updates.
          - sample batch from replay buffer
          - compute Q(s,a) with online network
          - compute TD target with frozen target network (no grad)
          - MSE loss, back propagation, clip gradients, optimizer step
          - hard-sync target every target_update_freq gradient steps
        """
        for agent in agents:
            policy: DQNPolicy = agent.policy
            policy.update_epsilon(agent.global_step)

            replay_buffer = agent.replay_buffer
            if (
                replay_buffer.size() < self.replay_start_size
                or agent.global_step % self.train_freq != 0
            ):
                continue

            optimizer = agent.aux_vars["optimizer"]

            for _ in range(self.grad_updates_per_step):
                batch = replay_buffer.sample_batch(self.batch_size)

                dev = policy.torch_device
                obs_t = iop.to_torch(batch["obs"], dev)
                next_obs_t = iop.to_torch(batch["next_obs"], dev)
                actions_t = iop.to_torch(batch["actions"], dev).long()
                rewards_t = iop.to_torch(batch["rewards"], dev)
                dones_t = iop.to_torch(batch["dones"], dev).float()

                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                if next_obs_t.dim() == 1:
                    next_obs_t = next_obs_t.unsqueeze(0)

                policy.q_network.torch_module.train()
                q_values = policy.q_network.torch_module(obs_t)          # [B, A]
                q_pred = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    policy.target_q_network.torch_module.eval()
                    next_q_values = policy.target_q_network.torch_module(next_obs_t)  # [B, A]
                    next_q_max, _ = next_q_values.max(dim=1)
                    td_target = rewards_t + self.gamma * (1.0 - dones_t) * next_q_max

                loss = F.mse_loss(q_pred, td_target)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(policy.parameters(), max_norm=10.0)
                optimizer.step()

                agent.train_step += 1
                agent.aux_vars["dqn_train_steps"] += 1

                if agent.aux_vars["dqn_train_steps"] % self.target_update_freq == 0:
                    policy.sync_target()
