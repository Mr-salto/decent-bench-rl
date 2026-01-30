from typing import Sequence, Dict, Any
import math

import numpy as np
import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import SupportedFrameworks, SupportedDevices

# PyTorch is required for this implementation
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required for QNetwork. Install torch in your environment.") from e


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


class QNetwork:
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
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for QNetwork")

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_sizes = list(hidden_sizes)

        self.torch_device = _resolve_torch_device(device)
        self._supported_device = _resolve_supported_device(device)

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
        )  # still works; trunk outputs features consumed by head, but for optimizer we reference parameters via .parameters()
        # Note: keeping trunk & head separate is convenient for introspection

        self._init_weights()
        self.to(self.torch_device)

    def _init_weights(self):
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

    def _to_torch_tensor(self, array):  # TO REWORK AND SHORTEN EVEN MORE
        """
        Convert an iop Array or native array to a torch tensor on the network device.
        """
        if isinstance(array, torch.Tensor):
            return arr.to(self.torch_device).float()

        t = iop.to_torch(array, self.torch_device)
        if isinstance(t, torch.Tensor):
            return t.float().to(self.torch_device)

    def _to_iop_array(self, tensor: torch.Tensor):
        """
        Convert torch.Tensor to an iop array via iop.to_array.
        We keep the tensor on CPU for safety when converting, but allow returning
        a device-aware array by passing the correct SupportedDevices.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected torch.Tensor as input to _to_iop_array")
        # detach and move to cpu for stable conversion; but we include device in call
        t_cpu = tensor.detach().cpu()
        arr = iop.to_array(t_cpu, SupportedFrameworks.TORCH, self._supported_device)
        return arr

    # -----------------------
    # Public API
    # -----------------------
    @property
    def torch_module(self) -> nn.Module:
        """Expose internal torch module for optimizer creation."""
        return nn.Sequential(self.trunk, self.head)

    def parameters(self):
        """Return iterator over torch parameters (convenient for optimizer)."""
        return self.torch_module.parameters()

    def to(self, device):
        """Move module to device. Accepts torch.device, str('cpu'/'cuda'), or SupportedDevices."""
        self.torch_device = _resolve_torch_device(device)
        self._supported_device = _resolve_supported_device(device)
        self.trunk.to(self.torch_device)
        self.head.to(self.torch_device)

    def forward(self, obs, *, deterministic: bool = True):
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
        if obs_t.dim() == 1:
            # single sample -> add batch dim
            obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            h = self.trunk(obs_t)
            out = self.head(h)  # shape [B, n_actions]

        return self._to_iop_array(out)

    __call__ = forward

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

    def copy_from(self, other: "QNetwork"):
        """Hard-copy parameters from another QNetwork instance (must be torch-backed)."""
        if not isinstance(other, QNetwork):
            raise TypeError("copy_from expects another QNetwork")
        self.torch_module.load_state_dict(other.torch_module.state_dict())
