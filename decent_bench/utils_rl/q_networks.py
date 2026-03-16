from typing import Sequence, Dict, Any, Callable
import math

import numpy as np
import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import SupportedFrameworks, SupportedDevices
from decent_bench.utils.array import Array

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required for QNetwork. Install torch in your environment.") from e

from torch.distributions import Categorical


def _resolve_supported_device(device) -> SupportedDevices:
    """
    Accept either:
      - SupportedDevices
      - torch.device
      - string like 'cpu' or 'cuda'
    Return SupportedDevices for interoperability conversions.
    """
    if device is None:
        return SupportedDevices.CPU
    if isinstance(device, SupportedDevices):
        return device
    if isinstance(device, torch.device):
        return SupportedDevices.GPU if device.type == "cuda" else SupportedDevices.CPU
    if isinstance(device, str):
        if "cuda" in device or "gpu" in device:
            return SupportedDevices.GPU
        return SupportedDevices.CPU
    return SupportedDevices.CPU


def _resolve_torch_device(device) -> torch.device:
    """Return a torch.device for the given input."""
    if isinstance(device, torch.device):
        return device
    if isinstance(device, SupportedDevices):
        return torch.device("cuda" if device == SupportedDevices.GPU else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return torch.device("cpu")


class BaseNetwork(nn.Module):
    """
    Base class for all networks using iop interface.
    Handles:
        - device resolution
        - iop <-> torch conversion
        - parameter exposure
    Subclasses must:
        - define self._module (nn.Module)
        - implement forward()
    """
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device: str | None = "cpu",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)

        self.torch_device = _resolve_torch_device(device)
        self._supported_device = _resolve_supported_device(device)
    
    @property
    def torch_module(self) -> nn.Module:
        """Expose internal torch module for optimizer creation."""
        return self._module
    
    def parameters(self):
        """Return iterator over torch parameters (convenient for optimizer)."""
        return self._module.parameters()
    
    def state_dict(self) -> Dict[str, Any]:
        """Return torch state dict (cpu tensors)."""
        sd = self.torch_module.state_dict()
        return {k: v.cpu().clone() for k, v in sd.items()}

    def load_state_dict(self, sd: Dict[str, Any]):
        """Load torch state dict (accepts numpy arrays or torch tensors)."""
        mapped = {}
        for k, v in sd.items():
            if isinstance(v, np.ndarray):
                mapped[k] = torch.tensor(v, device=self.torch_device)
            else:
                mapped[k] = v.to(self.torch_device) if isinstance(v, torch.Tensor) else v
        self.torch_module.load_state_dict(mapped)

    def copy_from(self, other: "BaseNetwork"):
        """Hard-copy parameters from another Network instance (must be torch-backed)."""
        if not isinstance(other, BaseNetwork):
            raise TypeError("copy_from expects another QNetwork")
        self.torch_module.load_state_dict(other.torch_module.state_dict())

    def to(self, device):
        """Move module to device. Accepts torch.device, str('cpu'/'cuda'), or SupportedDevices."""
        self.torch_device = _resolve_torch_device(device)
        self._supported_device = _resolve_supported_device(device)
        self._module.to(self.torch_device)
        
    def _to_torch_tensor(self, array): 
        """
        Convert an iop Array or native array to a torch tensor on the network device.
        """
        if isinstance(array, torch.Tensor):
            return array.to(self.torch_device).float()

        t = iop.to_torch(array, self.torch_device)
        if isinstance(t, torch.Tensor):
            return t.float().to(self.torch_device)

    def _to_iop_array(self, tensor: torch.Tensor) -> Array:
        """
        Convert torch.Tensor to an iop array via iop.to_array.
        We keep the tensor on CPU for safety when converting, but allow returning
        a device-aware array by passing the correct SupportedDevices.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected torch.Tensor as input to _to_iop_array")
        # detach and move to cpu for stable conversion; but we include device in call
        t_cpu = tensor.detach().cpu()
        arr = iop.to_array(t_cpu, SupportedFrameworks.PYTORCH, self._supported_device)
        return arr



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
        device: str | None = "cpu",
    ):
        super().__init__(
            obs_dim,
            n_actions,
            device,
        )

        self.hidden_sizes = list(hidden_sizes)

        layers = []
        in_dim = self.obs_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, self.n_actions)

        self._module = nn.Sequential(
            self.trunk, self.head
        )  
        # still works; trunk outputs features consumed by head, but for optimizer we reference parameters via .parameters()

        self._init_weights()
        self.to(self.torch_device)

    def _init_weights(self) -> None:
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(m.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.head.weight, a=math.sqrt(5))
        if self.head.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.head.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.head.bias, -bound, bound)

    def forward(self, obs: Array, *, deterministic: bool = True) -> Array:
        """
        Forward pass.

        Args:
            obs: iop Array or native array (shape [obs_dim] or [B, obs_dim])
            deterministic: unused here but kept for API compatibility
        Returns:
            q_values: iop-friendly array (runtime: torch.Tensor or framework-native).
                      Shape: [B, n_actions] or [n_actions] for single sample.
        """
        obs_t = self._to_torch_tensor(obs)
        with torch.no_grad():
            q_values_t = self.forward_torch(obs_t)
        return self._to_iop_array(q_values_t)

    # __call__ = forward

    def forward_torch(self, obs_t: torch.Tensor) -> torch.Tensor:
        """
        Torch-native forward. Accepts a torch.Tensor on self.torch_device and
        returns Q-values as a torch.Tensor with shape [B, n_actions] (or [1, n_actions]).
        Used for back propagation during training.
        """
        obs_t = obs_t.float()
        if not TORCH_AVAILABLE:
            raise RuntimeError("forward_torch requires PyTorch")
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        h = self.trunk(obs_t)       
        out = self.head(h)       
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

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        epsilon: float = 1.0,
        epsilon_schedule: Callable[[int], float] | None = None,
        device: str | None = "cpu",
    ):
        super().__init__(obs_dim, n_actions, device)
        self.epsilon = epsilon
        self.epsilon_schedule = epsilon_schedule

        self.q_network = QNetwork(obs_dim, n_actions, hidden_sizes, device)
        self.target_q_network = QNetwork(obs_dim, n_actions, hidden_sizes, device)
        self.target_q_network.copy_from(self.q_network)

        # Point _module to the online network's module so BaseNetwork helpers (parameters, state_dict, torch_module) operate on the online net only.
        self._module = self.q_network._module

    def parameters(self):
        """Return only online Q-network parameters (used by the optimizer)."""
        return self.q_network.parameters()

    def to(self, device) -> None:
        """Move both online and target networks to device."""
        self.torch_device = _resolve_torch_device(device)
        self._supported_device = _resolve_supported_device(device)
        self.q_network.to(device)
        self.target_q_network.to(device)

    def update_epsilon(self, step: int) -> None:
        """Decay epsilon via schedule if set."""
        if self.epsilon_schedule is not None:
            self.epsilon = float(self.epsilon_schedule(step))

    def sync_target(self) -> None:
        """Hard-copy online network weights into the target network."""
        self.target_q_network.copy_from(self.q_network)

    def get_action_and_value(self, obs: Array, deterministic: bool = False) -> tuple:
        """
        Epsilon-greedy action selection.

        Args:
            obs: observation (iop Array or numpy array), shape [obs_dim]
            deterministic: if True, always pick the greedy action (epsilon ignored)
        Returns:
            (action, None, q_value): action as int, no log-prob (None for DQN),
            and the Q-value of the selected action.
        """
        obs_t = self._to_torch_tensor(obs)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)  # [1, obs_dim]

        with torch.no_grad():
            self.q_network.torch_module.eval()
            q_values_t = self.q_network.torch_module(obs_t)  # [1, n_actions]

        if not deterministic and np.random.random() < self.epsilon:
            action = int(np.random.randint(self.n_actions))
        else:
            action = int(q_values_t.argmax(dim=1).item())

        q_value = float(q_values_t[0, action].item())
        return (action, None, q_value)

    def forward(self, obs: Array, *, deterministic: bool = True) -> Array:
        """Forward pass returning Q-values from the online network."""
        return self.q_network.forward(obs, deterministic=deterministic)


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
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        shared_features: bool = False,
        activation: type[nn.Module] = nn.Tanh, # choose right activation, can be sigmoid / ReLu
        device: str | None = "cpu",
    ):
        super().__init__(
            obs_dim,
            n_actions,
            device,
        )
        
        self.hidden_sizes = list(hidden_sizes)
        self.shared_features = shared_features
        self.activation = activation

        trunk, last_dim= self._build_mlp(self.obs_dim)
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
        self.to(self.torch_device)
    
    def _build_mlp(self, input_dim: int):
        layers = []
        in_dim = input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(self.activation())
            in_dim = h
        trunk = nn.Sequential(*layers)
        return trunk, in_dim

    # Orthogonal weight init could be added instead
    def _init_weights(self):
        for m in self._module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(m.bias, -bound, bound)

    def logits_values_torch(self, obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (logits, values) from torch input tensors.
        Args:
            obs_t: torch.Tensor shape [B, obs_dim] or [obs_dim]
        Returns:
            logits: [B, n_actions]
            values: [B]
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
    
    def forward_torch(self, obs: torch.Tensor, deterministic: bool = True) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """
        Forward pass in actor and critic networks. Used for training.

        Args:
            obs: Observations
            deterministic: Whether to sample or use deterministic actions
        Returns:
            action, value and log probability of the actions as torch tensors.
        """
        logits, values = self.logits_values_torch(obs)  # [B, A], [B]
        dist = Categorical(logits=logits) # create prob distribution associated with the current policy π(a|s)
        if deterministic:
            actions_t = torch.argmax(logits, dim=1)
        else:
            actions_t = dist.sample()  # [B]
        log_probs_t = dist.log_prob(actions_t)  # [B]
        # entropy = dist.entropy()  # optional: return if you want
        return actions_t, values, log_probs_t

    def forward(self, obs: Array, deterministic: bool = True) -> tuple[Array,Array,Array]:
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

    def evaluate_actions_torch(self, obs_t: torch.Tensor, actions_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute (log_probs, entropy, values) for given obs and actions based on the current policy.
        This function must allow gradients (used during policy update).
        Args:
            obs: Observations
            actions: actions whose probability is evaluated
        Returns:
            log likelihood of taking actions, entropy of the action distribution and estimated value
        """
        logits, values = self.logits_values_torch(obs_t)
        if actions_t.dim() == 0:
            actions_t = actions_t.unsqueeze(0) # [B]
        dist = Categorical(logits=logits)
        log_probs_t = dist.log_prob(actions_t)
        entropy_t = dist.entropy()
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
            obs: Observations
        Returns:
            estimated values as torch tensor
        """
        logits, values = self.logits_values_torch(obs_t)
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

    def get_action_and_value(self, obs: Array, deterministic: bool = False) -> tuple:
        """
        Interface compatible with RLAgent.act(). Equivalent to forward().

        Args:
            obs: observation (iop Array or numpy array), shape [obs_dim]
            deterministic: if True, take the greedy (argmax) action
        Returns:
            (action: int, logprob: float, value: float)
        """
        action_arr, value_arr, logprob_arr = self.forward(obs, deterministic=deterministic)
        action = int(iop.to_torch(action_arr, self._supported_device).squeeze().item())
        logprob = float(iop.to_torch(logprob_arr, self._supported_device).squeeze().item())
        value = float(iop.to_torch(value_arr, self._supported_device).squeeze().item())
        return (action, logprob, value)

    def get_distribution(self, obs: Array):
        """
        Get the action probability distribution with the current policy
        Args:
            obs: Observations
        Returns:
            distribution
        """
        obs_t = self._to_torch_tensor(obs)

        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        if self.shared_features:
            features = self.shared_trunk(obs_t)
            logits = self.policy_head(features)
        else:
            policy_feat = self.policy_trunk(obs_t)
            logits = self.policy_head(policy_feat)

        dist = Categorical(logits=logits)
        return dist


