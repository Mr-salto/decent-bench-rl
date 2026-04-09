from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, final

import numpy as np
import torch
from torch.nn import functional
from torch.nn.utils import clip_grad_norm_

import decent_bench.utils.interoperability as iop
from decent_bench.environments import PettingZooEnv
from decent_bench.rl_agents import LinearDecreasingEpsilon, RLAgent
from decent_bench.utils.types import SupportedDevices
from decent_bench.utils_rl.q_networks import ActorCritic, DQNPolicy
from decent_bench.utils_rl.replay_buffer import RolloutBuffer, SimpleReplayBuffer

StepResult = tuple[Any, float, bool, dict[str, Any]]


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

    def on_step_collect(self, _agents: list[RLAgent], _results: dict[RLAgent, StepResult]) -> None:
        """
        Call immediately after env.step(...).

        Used for on-policy collectors, logging, or any per-step bookkeeping.
        `results` is mapping agent -> (next_obs, reward, done, info).
        """
        return

    def on_step_update(self, _agents: list[RLAgent]) -> None:
        """
        Call once per environment step after on_step_collect.

        Used for off-policy updates.
        """
        return

    def on_episode_end(self, _agents: list[RLAgent]) -> None:
        """
        Call once when an episode ends (any agent done).

        Useful for on-policy algorithms
        that need to compute advantages and perform batch updates.
        """
        return

    def finalize(
        self,
        agents: list[RLAgent],
        env: PettingZooEnv,
    ) -> None:
        """
        Finalize the algorithm.

        Note:
            Override method as needed.
            Does not need to be implemented if no finalization is required.
            By default it is used to clean up auxiliary variables to free memory.

        Args:
            agents: list of agents.
            env: environment to run the algorithm on.

        """
        for i in agents:
            if i.aux_vars is not None:
                i.aux_vars.clear()
        env.close()

    @final
    def run(self, agents: list[RLAgent], env: PettingZooEnv) -> list[float]:
        """
        Run the algorithm.

        Note:
            This method first calls :meth:`initialize`, then runs the episode loop by repeatedly calling
            :meth:`on_step_collect`, :meth:`on_step_update` and :meth:`on_episode_end`. Finally calls :meth:`finalize`.
            Relies on transition collection to increment ``agent.global_step``.

        Warning:
            Do not override this method. Instead, override lifecycle hooks such as
            :meth:`initialize`, :meth:`on_step_collect`, :meth:`on_step_update`,
            :meth:`on_episode_end`, and :meth:`finalize` as needed.

        Args:
            agents: provides agents
            env: environment to run the algorithm on.

        Raises:
            RuntimeError: if any agent does not have an observation before action selection.

        """
        self.initialize(agents)
        mean_episode_returns: list[float] = []

        for _episode in range(self.episodes):
            action_dict: dict[RLAgent, int] = {}
            obs = env.reset()
            episode_done = False

            for _step in range(self.episode_length):
                for agent in agents:
                    if agent.latest_obs is None:
                        raise RuntimeError(f"Agent {agent.id} has no latest_obs before acting.")
                    obs[agent] = agent.latest_obs
                    action = agent.act(obs=obs[agent], deterministic=False)
                    action_dict[agent] = action

                results = env.step(action_dict)

                self.on_step_collect(agents, results)

                self.on_step_update(agents)

                if any(entry[2] for entry in results.values()):
                    per_agent_returns = []
                    for agent in agents:
                        if agent.episode_length > 0:
                            per_agent_returns.append(agent.finalize_episode())
                        elif agent.recent_returns:
                            per_agent_returns.append(agent.recent_returns[-1])
                        else:
                            per_agent_returns.append(agent.finalize_episode())
                    mean_episode_returns.append(float(np.mean(per_agent_returns)))
                    self.on_episode_end(agents)
                    episode_done = True
                    break

            # Append returns for episodes that end due to N_STEPS reaching max and not due to agents being done
            if not episode_done:
                per_agent_returns = []
                for agent in agents:
                    per_agent_returns.append(agent.finalize_episode())
                mean_episode_returns.append(float(np.mean(per_agent_returns)))
                self.on_episode_end(agents)

        self.finalize(agents, env)
        return mean_episode_returns


@dataclass(eq=False)
class IndependentDQN(RLAlgorithm):
    """
    Independent DQN (IDQN) algorithm — each agent trains its own Q-network using its local replay buffer.

    Each agent gets a DQNPolicy attached to agent.policy, which encapsulates the
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
    device: SupportedDevices = SupportedDevices.CPU

    def initialize(self, agents: list[RLAgent]) -> None:
        """
        Initialize agent-local DQN policies, optimizers, and replay buffers.

        Raises:
            RuntimeError: if an agent has no observation space.

        """
        for agent in agents:
            obs_space = agent.observation_space
            if obs_space is None:
                raise RuntimeError(f"Agent {agent.id} has no observation_space set during DQN.initialize()")
            obs_dim = int(np.prod(obs_space.shape))
            n_actions = agent.action_space.n

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
            agent.optimizer = opt
            agent.dqn_train_steps = 0

            if agent.replay_buffer is None:
                agent.replay_buffer = SimpleReplayBuffer(capacity=100000)

    def on_step_collect(self, _agents: list[RLAgent], results: dict[RLAgent, StepResult]) -> None:
        """
        IDQN collects transitions by calling agent.store_transition().

        Raises:
            RuntimeError: if any agent does not have an observation to add to the replay buffer.
            RuntimeError: if any agent does not have a last_action to add to the replay buffer.

        """
        for agent, (next_obs, reward, done, info) in results.items():
            prev_obs = agent.obs_at_act
            action = agent.last_action
            if prev_obs is None:
                raise RuntimeError(f"Agent {agent.id} has no obs_at_act during transition collection.")
            if action is None:
                raise RuntimeError(f"Agent {agent.id} has no last_action during transition collection.")
            agent.store_transition(prev_obs, action, reward, next_obs, done, info)

    def on_step_update(self, agents: list[RLAgent]) -> None:  # noqa: PLR0914
        """
        Train and updates parameters for each agent every `train_freq` steps using samples from their replay buffer.

        For each agent: decay epsilon, then (if buffer is ready) run gradient updates.
          - sample batch from replay buffer
          - compute Q(s,a) with online network
          - compute TD target with frozen target network (no grad)
          - MSE loss, back propagation, clip gradients, optimizer step
          - hard-sync target every target_update_freq gradient steps.

        Raises:
            RuntimeError: if any agent has no optimizer configured.

        """
        for agent in agents:
            policy: DQNPolicy = agent.policy
            policy.update_epsilon(agent.global_step)

            replay_buffer = agent.replay_buffer
            if replay_buffer.size() < self.replay_start_size or agent.global_step % self.train_freq != 0:
                continue

            optimizer = agent.optimizer
            if optimizer is None:
                raise RuntimeError(f"Agent {agent.id} has no optimizer configured.")

            for _ in range(self.grad_updates_per_step):
                batch = replay_buffer.sample_batch(self.batch_size)

                dev = policy.device
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
                q_values = policy.q_network.torch_module(obs_t)  # [B, A]
                q_pred = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    policy.target_q_network.torch_module.eval()
                    next_q_values = policy.target_q_network.torch_module(next_obs_t)  # [B, A]
                    next_q_max, _ = next_q_values.max(dim=1)
                    td_target = rewards_t + self.gamma * (1.0 - dones_t) * next_q_max

                loss = functional.mse_loss(q_pred, td_target)
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                clip_grad_norm_(policy.parameters(), max_norm=10.0)
                optimizer.step()

                agent.train_step += 1
                agent.dqn_train_steps += 1

                if agent.dqn_train_steps % self.target_update_freq == 0:
                    policy.sync_target()


@dataclass(eq=False)
class A2C(RLAlgorithm):
    """
    Independent A2C (Advantage Actor-Critic).

    Each agent trains its own ActorCritic network using its local on-policy rollout buffer.

    Algorithm type: synchronous, on-policy.
    Update trigger: end of each episode (full rollout).

    Update: single gradient step using:
      - Policy (actor) loss:  -E[log π(a|s) · A(s,a)]
      - Value (critic) loss:  MSE(V(s), return_t)
      - Entropy bonus:        -H(π)   (promotes exploration)

    Advantages are computed with Generalized Advantage Estimation (GAE).

    Hyperparameters:
      - gamma: discount factor
      - gae_lambda: GAE smoothing (1 = Monte-Carlo, 0 = 1-step TD)
      - lr: shared learning rate for actor and critic
      - vf_coef: value loss weight in the combined loss
      - ent_coef: entropy bonus weight
      - device: 'cpu' or 'cuda'
    """

    episodes: int = 100
    episode_length: int = 30
    name: str = "A2C"

    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    device: SupportedDevices = SupportedDevices.CPU

    def initialize(self, agents: list[RLAgent]) -> None:
        """
        Initialize agent-local ActorCritic policies and rollout buffers.

        Raises:
            RuntimeError: if an agent has no observation space.

        """
        for agent in agents:
            obs_space = agent.observation_space
            if obs_space is None:
                raise RuntimeError(f"Agent {agent.id} has no observation_space set during A2C.initialize()")
            obs_dim = int(np.prod(obs_space.shape))
            n_actions = agent.action_space.n

            policy = ActorCritic(
                obs_dim=obs_dim,
                n_actions=n_actions,
                hidden_sizes=(64, 64),
                shared_features=False,
                device=self.device,
            )
            agent.attach_policy(policy)

            optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)
            agent.optimizer = optimizer

            agent.rollout_buffer = RolloutBuffer()

    def on_step_collect(self, _agents: list[RLAgent], results: dict[RLAgent, StepResult]) -> None:
        """
        Store rollout transitions for each agent.

        This includes (obs_t, action_t, reward_t, done_t, logprob_t, value_t).

        obs_t is retrieved from agent.obs_at_act which was saved by RLAgent.act().
        This is the pre-step observation used to select the action.

        Raises:
            RuntimeError: if any agent does not have an observation to add to the rollout buffer.
            RuntimeError: if any agent does not have a last_action to add to the rollout buffer.

        """
        for agent, (next_obs, reward, done, _info) in results.items():
            obs_t = agent.obs_at_act
            if obs_t is None:
                obs_t = next_obs  # fallback: only possible if act() was not called
            if obs_t is None:
                raise RuntimeError(f"Agent {agent.id} has no observation to add to rollout buffer.")

            action_t = agent.last_action
            if action_t is None:
                raise RuntimeError(f"Agent {agent.id} has no last_action for rollout collection.")

            logprob_t = float(agent.last_logprob or 0.0)
            value_t = float(agent.last_value or 0.0)

            agent.rollout_buffer.add(obs_t, action_t, reward, done, logprob_t, value_t)

            # Update per-step episode tracking (mirrors store_transition for replay buffers)
            agent.global_step += 1
            agent.episode_return += float(reward)
            agent.episode_length += 1

            if done:
                agent.finalize_episode()

    def on_step_update(self, _agents: list[RLAgent]) -> None:
        """A2C is on-policy; gradient updates are deferred to on_episode_end."""
        return

    def on_episode_end(self, agents: list[RLAgent]) -> None:  # noqa: PLR0914
        """
        Train and update parameters for each agent at the end of the episode using the full rollout.

        1. Bootstrap the value of the last state.
        2. Compute GAE advantages and discounted returns.
        3. Single gradient step: policy loss + value loss + entropy bonus.
        4. Clear the rollout buffer for the next episode.
        """
        for agent in agents:
            rollout = agent.rollout_buffer
            if rollout.len() == 0:
                continue

            policy: ActorCritic = agent.policy
            optimizer = agent.optimizer
            dev = policy.device

            # Bootstrap: V(s_T) — if the last transition was terminal, GAE masks it out.
            last_obs = agent.latest_obs
            if last_obs is not None:
                with torch.no_grad():
                    last_obs_t = iop.to_torch(last_obs, dev)
                    last_value = float(policy.predict_values_torch(last_obs_t).squeeze().item())
            else:
                last_value = 0.0

            rollout.compute_returns_and_advantages(
                last_value=last_value,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )

            batch = rollout.get()

            obs_t = iop.to_torch(batch["obs"], dev)
            actions_t = iop.to_torch(batch["actions"], dev).long()
            advantages_t = iop.to_torch(batch["advantages"], dev)
            returns_t = iop.to_torch(batch["returns"], dev)

            # Normalise advantages for numerical stability
            if advantages_t.numel() > 1:
                advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            # --- Gradient update ---
            policy.torch_module.train()
            log_probs_t, entropy_t, values_t = policy.evaluate_actions_torch(obs_t, actions_t)

            policy_loss = -(log_probs_t * advantages_t.detach()).mean()
            value_loss = functional.mse_loss(values_t, returns_t)
            entropy_loss = -entropy_t.mean()

            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()

            agent.train_step += 1
            rollout.clear()
