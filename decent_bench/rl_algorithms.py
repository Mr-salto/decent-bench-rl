from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import final

import numpy as np
import torch
from torch.nn import functional
from torch.nn.utils import clip_grad_norm_

import decent_bench.utils.interoperability as iop
from decent_bench.environments import MPE, StepResult
from decent_bench.rl_agents import LinearDecreasingEpsilon, RLAgent
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices
from decent_bench.utils_rl.q_networks import ActorCritic, DQNPolicy, QMixer
from decent_bench.utils_rl.replay_buffer import JointReplayBuffer, RolloutBuffer, SimpleReplayBuffer


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
        env: MPE,
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
    def run(
        self,
        agents: list[RLAgent],
        env: MPE,
        episode_callback: Callable[[int], None] | None = None,
    ) -> list[float]:
        """
        Run the algorithm.

        Note:
            This method first calls :meth:`initialize`, then runs the episode loop by repeatedly calling
            :meth:`on_step_collect`, :meth:`on_step_update` and :meth:`on_episode_end`. Finally calls :meth:`finalize`.
            Relies on the environment to own generic step accounting and episode finalization.

        Warning:
            Do not override this method. Instead, override lifecycle hooks such as
            :meth:`initialize`, :meth:`on_step_collect`, :meth:`on_step_update`,
            :meth:`on_episode_end`, and :meth:`finalize` as needed.

        Args:
            agents: provides agents
            env: environment to run the algorithm on.
            episode_callback: optional callback invoked when an episode is finalized.
                Receives the 0-based episode index.

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

                results, episode_done, mean_episode_return = env.step(action_dict)

                self.on_step_collect(agents, results)

                self.on_step_update(agents)

                if episode_done:
                    if mean_episode_return is None:
                        raise RuntimeError("Environment returned episode_done=True without mean_episode_return.")
                    mean_episode_returns.append(float(mean_episode_return))
                    self.on_episode_end(agents)
                    if episode_callback is not None:
                        episode_callback(_episode)
                    episode_done = True
                    break

            # Append returns for episodes that end due to N_STEPS reaching max and not due to agents being done
            if not episode_done:
                per_agent_returns = [agent.finalize_episode() for agent in agents]
                mean_episode_returns.append(float(np.mean(per_agent_returns)))
                self.on_episode_end(agents)
                if episode_callback is not None:
                    episode_callback(_episode)

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
class QMIX(RLAlgorithm):
    """
    QMIX (centralized training with decentralized execution).

    Each agent has its own DQNPolicy. A mixing network combines per-agent Q-values
    into a joint Q_tot conditioned on the global state. Training is centralized via
    a shared replay buffer and a shared optimizer.

    Hyperparameters:
      - gamma: discount factor
      - lr: optimizer learning rate
      - batch_size: minibatch size
      - replay_start_size: minimal buffer size before training begins
      - target_update_freq: number of gradient updates between target hard-syncs
      - train_freq: perform a gradient update every `train_freq` env steps
      - grad_updates_per_step: gradient steps per training trigger
      - mixing_hidden_dim: mixer hidden size
      - hypernet_hidden_dim: hypernetwork hidden size
      - use_softplus: enforce positive mixing weights with softplus if True
      - reward_agg: team reward aggregation ("mean" or "sum")
    """

    episodes: int = 100
    episode_length: int = 30
    name: str = "QMIX"

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 32
    replay_start_size: int = 500
    target_update_freq: int = 100
    train_freq: int = 1
    grad_updates_per_step: int = 1
    mixing_hidden_dim: int = 32
    hypernet_hidden_dim: int = 64
    use_softplus: bool = True
    reward_agg: str = "mean"
    state_dim: int | None = None
    device: SupportedDevices = SupportedDevices.CPU
    qmix_mixer: QMixer | None = None
    qmix_target_mixer: QMixer | None = None

    def _initialize_mixer_from_state(self, agents: list[RLAgent], state: Array) -> None:
        if self.state_dim is None:
            shape = np.asarray(state).shape
            self.state_dim = int(np.prod(shape))

        mixer = QMixer(
            n_agents=len(agents),
            state_dim=self.state_dim,
            mixing_hidden_dim=self.mixing_hidden_dim,
            hypernet_hidden_dim=self.hypernet_hidden_dim,
            use_softplus=self.use_softplus,
            device=self.device,
        )
        target_mixer = QMixer(
            n_agents=len(agents),
            state_dim=self.state_dim,
            mixing_hidden_dim=self.mixing_hidden_dim,
            hypernet_hidden_dim=self.hypernet_hidden_dim,
            use_softplus=self.use_softplus,
            device=self.device,
        )
        target_mixer.copy_from(mixer)

        self.qmix_mixer = mixer
        self.qmix_target_mixer = target_mixer

        params: list[torch.nn.Parameter] = list(mixer.parameters())
        for agent in agents:
            policy = agent.policy
            params.extend(list(policy.parameters()))

        optimizer = torch.optim.Adam(params, lr=self.lr)
        for agent in agents:
            agent.optimizer = optimizer

    def initialize(self, agents: list[RLAgent]) -> None:
        """
        Initialize agent-local DQN policies, shared mixer, optimizer, and replay buffer.

        Raises:
            RuntimeError: if any agent has no observation space.

        """
        obs_dims: list[int] = []
        for agent in agents:
            obs_space = agent.observation_space
            if obs_space is None:
                raise RuntimeError(f"Agent {agent.id} has no observation_space set during QMIX.initialize()")
            obs_dim = int(np.prod(obs_space.shape))
            obs_dims.append(obs_dim)
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
            agent.dqn_train_steps = 0

        shared_replay = JointReplayBuffer(capacity=100000, n_agents=len(agents))

        self.qmix_mixer = None
        self.qmix_target_mixer = None

        for agent in agents:
            agent.replay_buffer = shared_replay
            agent.optimizer = None

    def on_step_collect(  # noqa: PLR0914
        self, agents: list[RLAgent], results: dict[RLAgent, StepResult]
    ) -> None:
        """
        Store joint transitions in the shared replay buffer.

        Raises:
            RuntimeError: if required agent state is missing.
            ValueError: if reward aggregation mode is unsupported.

        """
        replay_buffer = agents[0].replay_buffer
        if replay_buffer is None:
            raise RuntimeError("QMIX requires a shared replay buffer.")

        obs_list: list[Array] = []
        actions_list: list[int] = []
        rewards_list: list[float] = []
        next_obs_list: list[Array | None] = []
        done_flags: list[bool] = []
        global_state: Array | None = None
        next_global_state: Array | None = None

        for agent in agents:
            next_obs, reward, done, info = results[agent]
            if info and global_state is None:
                candidate_state = info["global_state"]
                candidate_next_state = info["next_global_state"]
                if candidate_state is not None and candidate_next_state is not None:
                    framework, device = iop.framework_device_of_array(candidate_state)
                    global_state = iop.to_array(candidate_state, framework, device)
                    next_global_state = iop.to_array(candidate_next_state, framework, device)

            prev_obs = agent.obs_at_act if agent.obs_at_act is not None else agent.latest_obs
            if prev_obs is None:
                raise RuntimeError(f"Agent {agent.id} has no obs_at_act for QMIX collection.")
            action = agent.last_action
            if action is None:
                raise RuntimeError(f"Agent {agent.id} has no last_action for QMIX collection.")

            obs_list.append(prev_obs)
            actions_list.append(int(action))
            rewards_list.append(float(reward))
            next_obs_list.append(next_obs)
            done_flags.append(bool(done))

        if self.reward_agg == "sum":
            team_reward = float(np.sum(rewards_list))
        elif self.reward_agg == "mean":
            team_reward = float(np.mean(rewards_list))
        else:
            raise ValueError(f"Unsupported reward_agg: {self.reward_agg}")

        episode_done = any(done_flags)

        if global_state is None or next_global_state is None:
            raise RuntimeError(
                "QMIX requires env-provided global_state/next_global_state. "
                "Ensure MPE.get_global_state() returns a state and is attached in info."
            )

        if self.qmix_mixer is None:
            self._initialize_mixer_from_state(agents, global_state)

        replay_buffer.add(
            obs=obs_list,
            actions=actions_list,
            reward=team_reward,
            next_obs=next_obs_list,
            done=episode_done,
            state=global_state,
            next_state=next_global_state,
            info=None,
        )

    def on_step_update(self, agents: list[RLAgent]) -> None:  # noqa: PLR0914, PLR0915
        """
        Update per-agent Q-networks and the mixer from the shared replay buffer.

        Raises:
            RuntimeError: if required replay buffer is missing.

        """
        replay_buffer = agents[0].replay_buffer
        if replay_buffer is None:
            raise RuntimeError("QMIX requires a shared replay buffer.")

        for agent in agents:
            agent.policy.update_epsilon(agent.global_step)

        if replay_buffer.size() < self.replay_start_size or agents[0].global_step % self.train_freq != 0:
            return

        mixer = self.qmix_mixer
        target_mixer = self.qmix_target_mixer
        optimizer = agents[0].optimizer
        if mixer is None or target_mixer is None or optimizer is None:
            return

        for _ in range(self.grad_updates_per_step):
            batch = replay_buffer.sample_batch(self.batch_size)

            dev = agents[0].policy.device
            obs_t = iop.to_torch(batch["obs"], dev)
            next_obs_t = iop.to_torch(batch["next_obs"], dev)
            actions_t = iop.to_torch(batch["actions"], dev).long()
            rewards_t = iop.to_torch(batch["rewards"], dev)
            dones_t = iop.to_torch(batch["dones"], dev).float()
            state_t = iop.to_torch(batch["state"], dev)
            next_state_t = iop.to_torch(batch["next_state"], dev)

            if obs_t.dim() == 2:
                obs_t = obs_t.unsqueeze(1)
                next_obs_t = next_obs_t.unsqueeze(1)
                actions_t = actions_t.unsqueeze(1)

            agent_qs: list[torch.Tensor] = []
            next_agent_qs: list[torch.Tensor] = []

            for idx, agent in enumerate(agents):
                policy = agent.policy
                policy.q_network.torch_module.train()
                q_all = policy.q_network.torch_module(obs_t[:, idx, ...])
                q_taken = q_all.gather(1, actions_t[:, idx].unsqueeze(1)).squeeze(1)
                agent_qs.append(q_taken)

                with torch.no_grad():
                    policy.target_q_network.torch_module.eval()
                    next_q_all = policy.target_q_network.torch_module(next_obs_t[:, idx, ...])
                    next_q_max, _ = next_q_all.max(dim=1)
                    next_agent_qs.append(next_q_max)

            agent_qs_t = torch.stack(agent_qs, dim=1)
            next_agent_qs_t = torch.stack(next_agent_qs, dim=1)

            mixer.torch_module.train()
            q_tot = mixer.forward_torch(agent_qs_t, state_t)

            with torch.no_grad():
                target_mixer.torch_module.eval()
                target_q_tot = target_mixer.forward_torch(next_agent_qs_t, next_state_t)
                td_target = rewards_t + self.gamma * (1.0 - dones_t) * target_q_tot

            loss = functional.mse_loss(q_tot, td_target)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]

            params = list(mixer.parameters())
            for agent in agents:
                params.extend(list(agent.policy.parameters()))
            clip_grad_norm_(params, max_norm=10.0)

            optimizer.step()

            for agent in agents:
                agent.train_step += 1
                agent.dqn_train_steps += 1

            if agents[0].dqn_train_steps % self.target_update_freq == 0:
                for agent in agents:
                    agent.policy.sync_target()
                target_mixer.copy_from(mixer)


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


@dataclass(eq=False)
class PPO(A2C):
    """
    Proximal Policy Optimization (PPO).

    PPO extends A2C by using a clipped surrogate objective and multiple
    epochs of minibatch updates over the same on-policy rollout.

    Hyperparameters (in addition to A2C):
      - clip_coef: PPO clip range (epsilon)
      - update_epochs: number of optimization epochs per rollout
      - minibatch_size: size of each minibatch (use full batch if <= 0)
    """

    name: str = "PPO"

    clip_coef: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 64

    def on_episode_end(self, agents: list[RLAgent]) -> None:  # noqa: PLR0914
        """
        Train and update parameters for each agent using PPO clipped updates.

        1. Bootstrap last value, compute GAE advantages and returns.
        2. Normalize advantages.
        3. Run multiple epochs of minibatch updates with clipped surrogate loss.
        4. Clear the rollout buffer.
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
            old_logprobs_t = iop.to_torch(batch["logprobs"], dev)

            # Normalise advantages for numerical stability
            if advantages_t.numel() > 1:
                advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            batch_size = int(advantages_t.shape[0])
            minibatch_size = int(self.minibatch_size)
            if minibatch_size <= 0 or minibatch_size > batch_size:
                minibatch_size = batch_size

            policy.torch_module.train()

            for _ in range(self.update_epochs):
                indices = torch.randperm(batch_size, device=advantages_t.device)

                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_idx = indices[start:end]

                    mb_obs = obs_t[mb_idx]
                    mb_actions = actions_t[mb_idx]
                    mb_adv = advantages_t[mb_idx]
                    mb_returns = returns_t[mb_idx]
                    mb_old_logprobs = old_logprobs_t[mb_idx]

                    log_probs_t, entropy_t, values_t = policy.evaluate_actions_torch(mb_obs, mb_actions)

                    ratio = torch.exp(log_probs_t - mb_old_logprobs)
                    unclipped = ratio * mb_adv
                    clipped = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv
                    policy_loss = -torch.min(unclipped, clipped).mean()

                    value_loss = functional.mse_loss(values_t, mb_returns)
                    entropy_loss = -entropy_t.mean()

                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()  # type: ignore[no-untyped-call]
                    clip_grad_norm_(policy.parameters(), max_norm=10.0)
                    optimizer.step()

                    agent.train_step += 1

            rollout.clear()
