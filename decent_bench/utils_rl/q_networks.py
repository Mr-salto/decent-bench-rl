from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from typing import cast

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

try:
    import torch
    from torch import nn
    from torch.nn import functional

    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required for QNetwork. Install torch in your environment.") from e

from torch.distributions import Categorical


def _resolve_torch_device(device: SupportedDevices) -> torch.device:
    """Map interoperability device enum to torch.device."""
    return torch.device("cuda" if device == SupportedDevices.GPU else "cpu")


class BaseNetwork(nn.Module):
    """
    Base class for all networks using iop interface.

    Handles:
        - device resolution
        - iop <-> torch conversion
        - parameter exposure
    Subclasses must:
        - define self._module (nn.Module)
        - implement forward().
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device: SupportedDevices = SupportedDevices.CPU,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = device
        self.torch_device = _resolve_torch_device(device)
        self._module: nn.Module = nn.Identity()

    @property
    def torch_module(self) -> nn.Module:
        """Expose internal torch module for optimizer creation."""
        return self._module

    def copy_from(self, other: BaseNetwork) -> None:
        """
        Hard-copy parameters from another Network instance (must be torch-backed).

        Raises:
            TypeError: if the other object is not a BaseNetwork.

        """
        if not isinstance(other, BaseNetwork):
            raise TypeError("copy_from expects another BaseNetwork")
        self.torch_module.load_state_dict(other.torch_module.state_dict())

    def move_to_device(self, device: SupportedDevices) -> None:
        """Move this network to the requested interoperability device."""
        self.device = device
        self.torch_device = _resolve_torch_device(device)
        self._module.to(self.torch_device)

    def _to_torch_tensor(self, array: Array | torch.Tensor) -> torch.Tensor:
        """
        Convert an iop Array or native array to a torch tensor on the network device.

        Raises:
            TypeError: If the input cannot be converted to a torch.Tensor.

        """
        if isinstance(array, torch.Tensor):
            return array.to(self.torch_device).float()

        t = iop.to_torch(array, self.device)
        if isinstance(t, torch.Tensor):
            return t.float().to(self.torch_device)
        raise TypeError("Could not convert input to torch.Tensor")

    def _to_iop_array(self, tensor: torch.Tensor) -> Array:
        """
        Convert torch.Tensor to an iop array via iop.to_array.

        We keep the tensor on CPU for safety when converting, but allow returning
        a device-aware array by passing the correct SupportedDevices.

        Raises:
            TypeError: If the input is not a torch.Tensor.

        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected torch.Tensor as input to _to_iop_array")
        # detach and move to cpu for stable conversion; but we include device in call
        t_cpu = tensor.detach().cpu()
        return iop.to_array(t_cpu, SupportedFrameworks.PYTORCH, self.device)


class QNetwork(BaseNetwork):
    """
    Torch-only Q-network with an iop interface.

    - Accepts iop Arrays (or native arrays) as input to forward()
    - Internally converts to torch.Tensor and runs a torch.nn.Module
    - Returns an iop array (at runtime a torch.Tensor) for Q-values

    Simple MLP (no dueling). Hidden sizes default to (64, 64).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        device: SupportedDevices = SupportedDevices.CPU,
    ) -> None:
        super().__init__(
            obs_dim,
            n_actions,
            device,
        )

        self.hidden_sizes = list(hidden_sizes)

        layers: list[nn.Module] = []
        in_dim = self.obs_dim
        for h in self.hidden_sizes:
            layers.extend((nn.Linear(in_dim, h), nn.ReLU(inplace=True)))
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, self.n_actions)

        self._module = nn.Sequential(self.trunk, self.head)
        # Trunk outputs features consumed by head; optimizer parameters come from the module tree.

        self._init_weights()
        self.move_to_device(self.device)

    def _init_weights(self) -> None:
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    fan_in = m.weight.shape[1]
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(m.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.head.weight, a=math.sqrt(5))
        if self.head.bias is not None:
            fan_in = self.head.weight.shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.head.bias, -bound, bound)

    def forward(self, obs: Array) -> Array:
        """
        Forward pass.

        Args:
            obs: iop Array or native array (shape [obs_dim] or [B, obs_dim])

        Returns:
            Q-values as iop array of shape [B, n_actions] or [n_actions] for single sample.

        """
        obs_t = self._to_torch_tensor(obs)
        with torch.no_grad():
            q_values_t = self.forward_torch(obs_t)
        return self._to_iop_array(q_values_t)

    def forward_torch(self, obs_t: torch.Tensor) -> torch.Tensor:
        """
        Torch-native forward.

        Accepts a torch.Tensor on self.torch_device and used for back propagation during training.

        Returns:
            Q-values as a torch.Tensor with shape [B, n_actions] (or [1, n_actions]).

        Raises:
            RuntimeError: if PyTorch is unavailable.
            TypeError: if the output of the forward pass is not a torch.Tensor.

        """
        obs_t = obs_t.float()
        if not TORCH_AVAILABLE:
            raise RuntimeError("forward_torch requires PyTorch")
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        out = self.torch_module(obs_t)
        if not isinstance(out, torch.Tensor):
            raise TypeError("QNetwork forward pass did not return a torch.Tensor")
        return out


class DQNPolicy(BaseNetwork):
    """
    DQN policy wrapping a QNetwork and a target QNetwork.

    Implements epsilon-greedy action selection via get_action_and_value(),
    which is the interface expected by RLAgent.act().

    Args:
        obs_dim: dimension of flattened observations
        n_actions: number of discrete actions
        hidden_sizes: MLP hidden layer sizes
        epsilon: initial exploration rate (1.0 = fully random)
        epsilon_schedule: optional callable (global_step -> epsilon) for decay
        device: execution device ('cpu' or 'cuda')

    """

    def __init__(  # noqa: PLR0917
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        epsilon: float = 1.0,
        epsilon_schedule: Callable[[int], float] | None = None,
        device: SupportedDevices = SupportedDevices.CPU,
    ) -> None:
        super().__init__(obs_dim, n_actions, device)
        self.epsilon = epsilon
        self.epsilon_schedule = epsilon_schedule

        self.q_network = QNetwork(obs_dim, n_actions, hidden_sizes, device)
        self.target_q_network = QNetwork(obs_dim, n_actions, hidden_sizes, device)
        self.target_q_network.copy_from(self.q_network)
        self._rng = np.random.default_rng()

        # BaseNetwork helpers operate on the online network module.
        self._module = self.q_network.torch_module

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return only online Q-network parameters (used by the optimizer)."""
        return self.q_network.parameters(recurse=recurse)

    def move_to_device(self, device: SupportedDevices) -> None:
        """Move both online and target networks to device."""
        super().move_to_device(device)
        self.q_network.move_to_device(device)
        self.target_q_network.move_to_device(device)

    def update_epsilon(self, step: int) -> None:
        """Decay epsilon via schedule if set."""
        if self.epsilon_schedule is not None:
            self.epsilon = float(self.epsilon_schedule(step))

    def sync_target(self) -> None:
        """Hard-copy online network weights into the target network."""
        self.target_q_network.copy_from(self.q_network)

    def get_action_and_value(self, obs: Array, deterministic: bool = False) -> tuple[int, None, float]:
        """
        Epsilon-greedy action selection.

        Args:
            obs: observation (iop Array or numpy array), shape [obs_dim]
            deterministic: if True, always pick the greedy action (epsilon ignored)

        Returns:
            action as int, no log-prob (None for DQN), and the Q-value of the selected action.

        """
        obs_t = self._to_torch_tensor(obs)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)  # [1, obs_dim]

        with torch.no_grad():
            self.q_network.torch_module.eval()
            q_values_t = self.q_network.torch_module(obs_t)  # [1, n_actions]

        if not deterministic and self._rng.random() < self.epsilon:
            action = int(self._rng.integers(self.n_actions))
        else:
            action = int(q_values_t.argmax(dim=1).item())

        q_value = float(q_values_t[0, action].item())
        return (action, None, q_value)

    def forward(self, obs: Array) -> Array:
        """Forward pass returning Q-values from the online network."""
        return self.q_network.forward(obs)


class QMixer(BaseNetwork):
    """
    QMIX mixing network with hypernet-generated weights conditioned on global state.

    Combines per-agent Q-values into a joint Q_tot while enforcing monotonicity
    w.r.t. each agent utility via non-negative mixing weights.
    """

    def __init__(  # noqa: PLR0917
        self,
        n_agents: int,
        state_dim: int,
        mixing_hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
        device: SupportedDevices = SupportedDevices.CPU,
        use_softplus: bool = True,
    ) -> None:
        super().__init__(obs_dim=state_dim, n_actions=n_agents, device=device)
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.mixing_hidden_dim = int(mixing_hidden_dim)
        self.hypernet_hidden_dim = int(hypernet_hidden_dim)
        self.use_softplus = bool(use_softplus)

        self.hyper_w1 = self._make_hypernet(
            out_dim=self.n_agents * self.mixing_hidden_dim,
            hidden_dim=self.hypernet_hidden_dim,
        )
        self.hyper_b1 = self._make_hypernet(
            out_dim=self.mixing_hidden_dim,
            hidden_dim=self.hypernet_hidden_dim,
        )
        self.hyper_w2 = self._make_hypernet(
            out_dim=self.mixing_hidden_dim,
            hidden_dim=self.hypernet_hidden_dim,
        )
        self.hyper_b2 = self._make_hypernet(
            out_dim=1,
            hidden_dim=self.hypernet_hidden_dim,
        )

        self._module = nn.ModuleDict({
            "hyper_w1": self.hyper_w1,
            "hyper_b1": self.hyper_b1,
            "hyper_w2": self.hyper_w2,
            "hyper_b2": self.hyper_b2,
        })

        self.move_to_device(self.device)

    def _make_hypernet(self, out_dim: int, hidden_dim: int) -> nn.Module:
        if hidden_dim <= 0:
            return nn.Linear(self.state_dim, out_dim)
        return nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def _enforce_nonneg(self, w: torch.Tensor) -> torch.Tensor:
        if self.use_softplus:
            return functional.softplus(w)
        return torch.abs(w)

    def forward_torch(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Compute joint Q_tot from per-agent Q-values and global state.

        Args:
            agent_qs: torch.Tensor shape [B, n_agents] or [n_agents]
            states: torch.Tensor shape [B, state_dim] or [state_dim]

        Returns:
            Q_tot as torch.Tensor with shape [B] (or scalar for single sample).

        """
        if agent_qs.dim() == 1:
            agent_qs = agent_qs.unsqueeze(0)
        if states.dim() == 1:
            states = states.unsqueeze(0)

        batch_size = agent_qs.shape[0]
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)

        w1 = self._enforce_nonneg(self.hyper_w1(states))
        w1 = w1.view(batch_size, self.n_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.mixing_hidden_dim)

        hidden = functional.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = self._enforce_nonneg(self.hyper_w2(states))
        w2 = w2.view(batch_size, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        q_tot_t = cast("torch.Tensor", q_tot)
        return q_tot_t.view(-1)

    def forward(self, agent_qs: Array, states: Array) -> Array:
        """Forward pass (iop arrays), intended for inference."""
        agent_qs_t = self._to_torch_tensor(agent_qs)
        states_t = self._to_torch_tensor(states)
        with torch.no_grad():
            q_tot_t = self.forward_torch(agent_qs_t, states_t)
        return self._to_iop_array(q_tot_t)


class ActorCritic(BaseNetwork):
    """
    ActorCritic model that produces both policy and value outputs.

    Modes:
      - Independent: actor and critic each have a separate network (default)
      - shared_features (actor and critic share a common trunk)

    Args:
        obs_dim (int): dimension of flattened observations
        n_actions (int): number of discrete actions
        hidden_sizes (tuple[int, ...]): sizes of MLP hidden layers
        share_features (bool): whether to use a shared trunk
        activation (Callable): activation function for hidden layers
        device (str | torch.device): execution device ('cpu' or 'cuda')

    """

    def __init__(  # noqa: PLR0917
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        shared_features: bool = False,
        activation: type[nn.Module] = nn.Tanh,  # choose right activation, can be sigmoid / ReLu
        device: SupportedDevices = SupportedDevices.CPU,
    ) -> None:
        super().__init__(
            obs_dim,
            n_actions,
            device,
        )

        self.hidden_sizes = list(hidden_sizes)
        self.shared_features = shared_features
        self.activation = activation

        trunk, last_dim = self._build_mlp(self.obs_dim)
        if self.shared_features:
            self.policy_trunk = trunk
            self.value_trunk = trunk
            self.policy_head = nn.Linear(last_dim, self.n_actions)
            self.value_head = nn.Linear(last_dim, 1)
        else:
            self.policy_trunk = trunk
            self.value_trunk, _ = self._build_mlp(self.obs_dim)
            self.policy_head = nn.Linear(last_dim, self.n_actions)
            self.value_head = nn.Linear(last_dim, 1)

        self._module = nn.ModuleDict({
            "policy_trunk": self.policy_trunk,
            "policy_head": self.policy_head,
            "value_trunk": self.value_trunk,
            "value_head": self.value_head,
        })

        self._init_weights()
        self.move_to_device(self.device)

    def _build_mlp(self, input_dim: int) -> tuple[nn.Sequential, int]:
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in self.hidden_sizes:
            layers.extend((nn.Linear(in_dim, h), self.activation()))
            in_dim = h
        trunk = nn.Sequential(*layers)
        return trunk, in_dim

    # Orthogonal weight init could be added instead
    def _init_weights(self) -> None:
        for m in self._module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    fan_in = m.weight.shape[1]
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(m.bias, -bound, bound)

    def logits_values_torch(self, obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (logits, values) from torch input tensors.

        Args:
            obs_t: torch.Tensor shape [B, obs_dim] or [obs_dim]

        Returns:
            logits with shape [B, n_actions]
            values with shape [B]

        """
        obs_t = obs_t.float()
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        if self.shared_features:
            features = self.policy_trunk(obs_t)
            logits = self.policy_head(features)
            values = self.value_head(features).squeeze(-1)
        else:
            policy_feat = self.policy_trunk(obs_t)
            value_feat = self.value_trunk(obs_t)
            logits = self.policy_head(policy_feat)
            values = self.value_head(value_feat).squeeze(-1)

        return logits, values

    def forward_torch(
        self, obs: torch.Tensor, deterministic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in actor and critic networks. Used for training.

        Args:
            obs: Observations
            deterministic: Whether to sample or use deterministic actions
        Returns:
            action, value and log probability of the actions as torch tensors.

        """
        logits, values = self.logits_values_torch(obs)  # [B, A], [B]
        dist = Categorical(logits=logits)  # create prob distribution associated with the current policy π(a|s)
        actions_t = torch.argmax(logits, dim=1) if deterministic else dist.sample()  # type: ignore[no-untyped-call]
        log_probs_t = dist.log_prob(actions_t)  # type: ignore[no-untyped-call] # [B]
        return actions_t, values, log_probs_t

    def forward(self, obs: Array, deterministic: bool = True) -> tuple[Array, Array, Array]:
        """
        Forward pass in actor and critic networks.

        Args:
            obs: Observations
            deterministic: Whether to sample or use deterministic actions
        Returns:
            action, value and log probability of the actions as iop Arrays

        """
        obs_t = self._to_torch_tensor(obs)  # shape: [obs_dim] or [B, obs_dim]
        with torch.no_grad():
            actions_t, values_t, log_probs_t = self.forward_torch(obs_t, deterministic=deterministic)
        action_arr = self._to_iop_array(actions_t)
        value_arr = self._to_iop_array(values_t)
        logprob_arr = self._to_iop_array(log_probs_t)

        return action_arr, value_arr, logprob_arr

    def evaluate_actions_torch(
        self, obs_t: torch.Tensor, actions_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute (log_probs, entropy, values) for given obs and actions based on the current policy.

        This function must allow gradients (used during policy update).

        Args:
            obs_t: Observations as Torch.tensor
            actions_t: actions whose probability is evaluated as Torch.tensor
        Returns:
            log likelihood of taking actions, entropy of the action distribution and estimated value

        """
        logits, values = self.logits_values_torch(obs_t)
        if actions_t.dim() == 0:
            actions_t = actions_t.unsqueeze(0)  # [B]
        dist = Categorical(logits=logits)
        log_probs_t = dist.log_prob(actions_t)  # type: ignore[no-untyped-call]
        entropy_t = dist.entropy()  # type: ignore[no-untyped-call]
        return log_probs_t, entropy_t, values

    def evaluate_actions(self, obs: Array, actions: Array) -> tuple[Array, Array, Array]:
        """
        Compute (log_probs, entropy, values) for given obs and actions based on the current policy.

        This function must allow gradients (used during policy update).

        Args:
            obs: Observations
            actions: actions whose probability is evaluated
        Returns:
            log likelihood of taking actions, entropy of the action distribution and estimated value

        """
        obs_t = self._to_torch_tensor(obs)
        actions_t = self._to_torch_tensor(actions).long()

        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if actions_t.dim() == 0:
            actions_t = actions_t.unsqueeze(0)

        logprobs_t, entropy_t, values_t = self.evaluate_actions_torch(obs_t, actions_t)

        logprobs_arr = self._to_iop_array(logprobs_t)
        entropy_arr = self._to_iop_array(entropy_t)
        values_arr = self._to_iop_array(values_t)

        return logprobs_arr, entropy_arr, values_arr

    def predict_values_torch(self, obs_t: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        Args:
            obs_t: Observations as Torch.tensor
        Returns:
            estimated values as torch tensor

        """
        _logits, values = self.logits_values_torch(obs_t)
        return values

    def predict_values(self, obs: Array) -> Array:
        """
        Get the estimated values according to the current policy given the observations.

        Args:
            obs: Observations

        Returns:
            estimated values as iop Array

        """
        obs_t = self._to_torch_tensor(obs)
        with torch.no_grad():
            values_t = self.predict_values_torch(obs_t)
        return self._to_iop_array(values_t)

    def get_action_and_value(self, obs: Array, deterministic: bool = False) -> tuple[int, float, float]:
        """
        Interface compatible with RLAgent.act(). Equivalent to forward().

        Args:
            obs: observation (iop Array or numpy array), shape [obs_dim]
            deterministic: if True, take the greedy (argmax) action

        Returns:
            action, log-probability, value estimate.

        """
        action_arr, value_arr, logprob_arr = self.forward(obs, deterministic=deterministic)
        action = int(iop.to_torch(action_arr, self.device).squeeze().item())
        logprob = float(iop.to_torch(logprob_arr, self.device).squeeze().item())
        value = float(iop.to_torch(value_arr, self.device).squeeze().item())
        return (action, logprob, value)

    def get_distribution(self, obs: Array) -> Categorical:
        """
        Get the action probability distribution with the current policy.

        Args:
            obs: Observations

        Returns:
            distribution.

        """
        obs_t = self._to_torch_tensor(obs)

        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        policy_features = self.policy_trunk(obs_t)
        logits = self.policy_head(policy_features)
        return Categorical(logits=logits)
