from dataclasses import dataclass, field
from typing import Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.normalize import Normalize, Unnormalize
from torch import nn, Tensor
import torch
from lerobot.common.policies.flowbc.configuration_flowbc import FlowbcConfig


class FlowbcPolicy(PreTrainedPolicy):
    """Torch implementation of the FlowBC Policy."""

    config_class = FlowbcConfig
    name = "flowbc"

    def __init__(self, config: FlowbcConfig, dataset_stats: Optional[dict] = None):
        super().__init__(config)
        self.config = config

        self.normalize_inputs = Normalize({"observation.state": config.ob_dims}, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize({"action": (config.action_dim,)}, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize({"action": (config.action_dim,)}, config.normalization_mapping, dataset_stats)

        # NOTE: Here you should define actor_bc_flow and actor_onestep_flow modules (simple MLPs or vector fields)
        self.actor_bc_flow = self._build_mlp(config.actor_hidden_dims, config.action_dim)
        self.actor_onestep_flow = self._build_mlp(config.actor_hidden_dims, config.action_dim)

    def _build_mlp(self, hidden_dims, output_dim):
        layers = []
        input_dim = self.config.ob_dims[0]  # Assume flat input for now
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            if self.config.actor_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self.normalize_inputs(batch)
        obs = batch["observation.state"]
        actions = batch["action"]

        # Compute flow-based velocity
        batch_size = obs.shape[0]
        x_0 = torch.randn_like(actions)
        t = torch.rand(batch_size, 1, device=actions.device)
        x_t = (1 - t) * x_0 + t * actions
        vel = actions - x_0

        pred = self.actor_bc_flow(torch.cat([obs, x_t, t], dim=-1))
        bc_flow_loss = ((pred - vel) ** 2).mean()

        with torch.no_grad():
            target_actions = self.compute_flow_actions(obs)
        actor_actions = self.actor_onestep_flow(torch.cat([obs, torch.randn_like(actions)], dim=-1))
        distill_loss = ((actor_actions - target_actions) ** 2).mean()

        actor_loss = bc_flow_loss + self.config.alpha * distill_loss
        return actor_loss, None

    def compute_flow_actions(self, obs: Tensor) -> Tensor:
        actions = torch.randn(obs.shape[0], self.config.action_dim, device=obs.device)
        for i in range(self.config.flow_steps):
            t = torch.full((obs.shape[0], 1), i / self.config.flow_steps, device=obs.device)
            input = torch.cat([obs, actions, t], dim=-1)
            vel = self.actor_bc_flow(input)
            actions = actions + vel / self.config.flow_steps
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        batch = self.normalize_inputs(batch)
        obs = batch["observation.state"]
        noise = torch.randn(obs.shape[0], self.config.action_dim, device=obs.device)
        input = torch.cat([obs, noise], dim=-1)
        action = self.actor_onestep_flow(input)
        action = torch.clamp(action, -1, 1)
        action = self.unnormalize_outputs({"action": action})["action"]
        return action
