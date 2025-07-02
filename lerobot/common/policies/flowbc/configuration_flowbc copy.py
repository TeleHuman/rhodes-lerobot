
from dataclasses import dataclass, field
from typing import Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.optim.optimizers import AdamConfig


@PreTrainedConfig.register_subclass("flowbc")
@dataclass
class FlowbcConfig(PreTrainedConfig):
    """Configuration class for Flow-based Behavior Cloning (FlowBC)."""
    # Model architecture
    actor_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    value_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    layer_norm: bool = True
    actor_layer_norm: bool = False

    # Training hyperparameters
    lr: float = 3e-4
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    q_agg: str = "mean"  # or "min"
    alpha: float = 10.0
    flow_steps: int = 10
    normalize_q_loss: bool = False

    # Encoder name if used
    encoder: Optional[str] = None

    # Dimensions (filled at runtime from dataset)
    ob_dims: Optional[tuple[int, ...]] = None
    action_dim: Optional[int] = None

    # Input/output normalization modes
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    def __post_init__(self):
        super().__post_init__()

        if self.q_agg not in ["mean", "min"]:
            raise ValueError(f"Unsupported q_agg: {self.q_agg}, expected 'mean' or 'min'")

        if self.ob_dims is None or self.action_dim is None:
            raise ValueError("You must provide `ob_dims` and `action_dim` from dataset or environment spec.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.lr,
            weight_decay=0.0  # Optional: adjust if needed
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return [0]

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.ob_dims is None or self.action_dim is None:
            raise ValueError("Both observation and action dimensions must be set.")

    @property
    def input_shapes(self) -> dict:
        return {"observation.state": list(self.ob_dims)} if self.ob_dims else {}

    @property
    def output_shapes(self) -> dict:
        return {"action": [self.action_dim]} if self.action_dim else {}
