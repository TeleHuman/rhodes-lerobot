#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
from torch import Tensor, nn
from typing import Union
from collections import defaultdict

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


def create_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    """
    Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
    statistics.

    Args: (see Normalize and Unnormalize)

    Returns:
        dict: A dictionary where keys are modalities and values are `nn.ParameterDict` containing
            `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.
    """
    stats_buffers = {}

    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        assert isinstance(norm_mode, NormalizationMode)

        shape = tuple(ft.shape)

        if ft.type is FeatureType.VISUAL:
            # sanity checks
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            # override image shape to be invariant to height and width
            shape = (c, 1, 1)

        # Note: we initialize mean, std, min, max to infinity. They should be overwritten
        # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
        # we assert they are not infinity anymore.

        buffer = {}
        if norm_mode is NormalizationMode.MEAN_STD:
            mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.MIN_MAX:
            min = torch.ones(shape, dtype=torch.float32) * torch.inf
            max = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(min, requires_grad=False),
                    "max": nn.Parameter(max, requires_grad=False),
                }
            )

        # TODO(aliberts, rcadene): harmonize this to only use one framework (np or torch)
        if stats:
            if isinstance(stats[key]["mean"], np.ndarray):
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = torch.from_numpy(stats[key]["mean"]).to(dtype=torch.float32)
                    buffer["std"].data = torch.from_numpy(stats[key]["std"]).to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = torch.from_numpy(stats[key]["min"]).to(dtype=torch.float32)
                    buffer["max"].data = torch.from_numpy(stats[key]["max"]).to(dtype=torch.float32)
            elif isinstance(stats[key]["mean"], torch.Tensor):
                # Note: The clone is needed to make sure that the logic in save_pretrained doesn't see duplicated
                # tensors anywhere (for example, when we use the same stats for normalization and
                # unnormalization). See the logic here
                # https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L97.
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = stats[key]["mean"].clone().to(dtype=torch.float32)
                    buffer["std"].data = stats[key]["std"].clone().to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = stats[key]["min"].clone().to(dtype=torch.float32)
                    buffer["max"].data = stats[key]["max"].clone().to(dtype=torch.float32)
            else:
                type_ = type(stats[key]["mean"])
                raise ValueError(f"np.ndarray or torch.Tensor expected, but type is '{type_}' instead.")

        stats_buffers[key] = buffer
    return stats_buffers


def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )


class Normalize(nn.Module):
    """Normalizes data (e.g. "observation.image") for more stable and faster convergence during training."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                # FIXME(aliberts, rcadene): This might lead to silent fail!
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = (batch[key] - mean) / (std + 1e-8)
            elif norm_mode is NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                # normalize to [0,1]
                batch[key] = (batch[key] - min) / (max - min + 1e-8)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            else:
                raise ValueError(norm_mode)
        return batch
    

def convert_stat_value(
        value: Union[np.ndarray, torch.Tensor], 
    ) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).to(dtype=torch.float32)
        elif isinstance(value, torch.Tensor):
            tensor = value.clone().to(dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported type: {type(value)}. Expected np.ndarray or torch.Tensor.")
        
        return tensor
    
def create_multi_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, dict[str, Tensor]]],
) -> dict[str, dict[str, nn.ParameterDict]]:
    """
    Create buffers per dataset (e.g. 'bridge', 'rt1') per modality (e.g. "observation.image", "action") containing their 
    mean, std, min, max statistics.

    Args: (see MultiDatasetNormalize and MultiDatasetUnnormalize)

    Returns:
        dict: A dictionary where keys are dataset name and values are dictionaries. In each dataset dictionary,  keys are modalities 
        and values are `nn.ParameterDict` containing `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.
    """

    stats_buffers = {}
    
    for dataset, ds_stats in stats.items():
        ds_buffers = {}

        for key, ft in features.items():
            norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue
            
            assert isinstance(norm_mode, NormalizationMode)
            if key not in ds_stats:
                raise ValueError(f"Feature '{key}' missing in stats for dataset '{dataset}'")
            
            shape = tuple(ft.shape)
            
            if ft.type is FeatureType.VISUAL:
                # sanity checks
                assert len(shape) == 3, f"Visual feature {key} should have 3 dims, got {shape}"
                c, h, w = shape
                assert c < h and c < w, f"{key} is not channel first ({shape=})"
                # override image shape to be invariant to height and width
                shape = (c, 1, 1)
            
            # Note: we initialize mean, std, min, max to infinity. They should be overwritten
            # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
            # we assert they are not infinity anymore.

            buffer = {}
            if norm_mode is NormalizationMode.MEAN_STD:
                mean = torch.ones(shape, dtype=torch.float32) * torch.inf
                std = torch.ones(shape, dtype=torch.float32) * torch.inf
                buffer = nn.ParameterDict(
                    {
                        "mean": nn.Parameter(mean, requires_grad=False),
                        "std": nn.Parameter(std, requires_grad=False),
                    }
                )
                
            elif norm_mode is NormalizationMode.MIN_MAX:
                min = torch.ones(shape, dtype=torch.float32) * torch.inf
                max = torch.ones(shape, dtype=torch.float32) * torch.inf
                buffer = nn.ParameterDict(
                    {
                        "min": nn.Parameter(min, requires_grad=False),
                        "max": nn.Parameter(max, requires_grad=False),
                    }
                )

            if ds_stats:
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = convert_stat_value(ds_stats[key]["mean"])
                    buffer["std"].data = convert_stat_value(ds_stats[key]["std"])
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = convert_stat_value(ds_stats[key]["min"])
                    buffer["max"].data = convert_stat_value(ds_stats[key]["max"])
            
            ds_buffers[key] = buffer

        stats_buffers[dataset] = ds_buffers
            
    return stats_buffers

class MultiDatasetNormalize(nn.Module):
    """Normalizes data from multiple with their respective statistics."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, dict[str, Tensor]]] | None = None,
    ):
        """
        Args:
            stats (dict, optional): A dictionary of statistics for multiple datasets, where the keys are dataset names (repo_id)
                and the values are dictionaries. In each dataset dictionary, keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.

        Examples:
            {
                'IPEC-COMMUNITY/bridge_orig_lerobot': {
                    'observation.state': {
                        'mean': ...,
                        'std': ...,
                        ...
                    },
                    ...
                },
                ...
            }
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        stats_buffers = create_multi_stats_buffers(features, norm_map, stats)
        for dataset_idx, ds_buffers in stats_buffers.items():
            for key, buffer in ds_buffers.items():
                # Create a unique name and register as submodule
                name = f"buffer_{dataset_idx}_{key.replace('.', '_')}"
                setattr(self, name, buffer)
    
    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        dataset_idx = batch.pop("dataset_index")     # repo_id

        for key, ft in self.features.items():
            if key not in batch:
                # FIXME(aliberts, rcadene): This might lead to silent fail!
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue
            try:
                buffer = getattr(self, "buffer_" + str(dataset_idx.item()) + "_" + key.replace(".", "_"))
            except KeyError:
                available = ", ".join(self.stats_buffers.keys())
                raise KeyError(
                    f"No stats for dataset index'{dataset_idx}' and feature '{key}'. "
                    f"Available datasets: {available}"
                ) from None

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str(f"mean for {key} in {dataset_idx}")
                assert not torch.isinf(std).any(), _no_stats_error_str(f"std for {key} in {dataset_idx}")
                batch[key] = (batch[key] - mean) / (std + 1e-8)
            elif norm_mode is NormalizationMode.MIN_MAX:
                min_val, max_val = buffer["min"], buffer["max"]
                assert not torch.isinf(min_val).any(), _no_stats_error_str(f"min for {key} in {dataset_idx}")
                assert not torch.isinf(max_val).any(), _no_stats_error_str(f"max for {key} in {dataset_idx}")
                # normalize to [0,1]
                batch[key] = (batch[key] - min) / (max - min + 1e-8)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            else:
                raise ValueError(norm_mode)

        return batch



class Unnormalize(nn.Module):
    """
    Similar to `Normalize` but unnormalizes output data (e.g. `{"action": torch.randn(b,c)}`) in their
    original range used by the environment.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        # `self.buffer_observation_state["mean"]` contains `torch.tensor(state_dim)`
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = batch[key] * std + mean
            elif norm_mode is NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max - min) + min
            else:
                raise ValueError(norm_mode)
        return batch


class MultiDatasetUnnormalize(nn.Module):
    """Similar to `MultiDatasetNormalize` but unnormalizes output data (e.g. `{"action": torch.randn(b,c)}`) in their
    original range used by the environment."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, dict[str, Tensor]]] | None = None,
    ):
        """
        Args:
            stats (dict, optional): A dictionary of statistics for multiple datasets, where the keys are dataset names (repo_id)
                and the values are dictionaries. In each dataset dictionary, keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.

        Examples:
            {
                'IPEC-COMMUNITY/bridge_orig_lerobot': {
                    'observation.state': {
                        'mean': ...,
                        'std': ...,
                        ...
                    },
                    ...
                },
                ...
            }
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        stats_buffers = create_multi_stats_buffers(features, norm_map, stats)
        for dataset_idx, ds_buffers in stats_buffers.items():
            for key, buffer in ds_buffers.items():
                # Create a unique name and register as submodule
                name = f"buffer_{dataset_idx}_{key.replace('.', '_')}"
                setattr(self, name, buffer)
    
    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        data_sources = batch.pop("dataset") 

        # Group indices by dataset
        source_indices = defaultdict(list)
        for i, source in enumerate(data_sources):
            source_indices[source].append(i)

        for key, ft in self.features.items():
            if key not in batch:
                # FIXME(aliberts, rcadene): This might lead to silent fail!
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            x = batch[key]
            for dataset_idx, indices in source_indices.items():
                # Skip if no data for this source
                if not indices:
                    continue
                try:
                    buffer = getattr(self, "buffer_" + dataset_idx + "_" + key.replace(".", "_"))
                except KeyError:
                    available = ", ".join(self.stats_buffers.keys())
                    raise KeyError(
                        f"No stats for dataset idx '{dataset_idx}' and feature '{key}'. "
                        f"Available datasets: {available}"
                    ) from None
                
                # Extract the slice of data for this source
                x_slice = x[indices]

                if norm_mode is NormalizationMode.MEAN_STD:
                    mean = buffer["mean"]
                    std = buffer["std"]
                    assert not torch.isinf(mean).any(), _no_stats_error_str(f"mean for {key} in {source}")
                    assert not torch.isinf(std).any(), _no_stats_error_str(f"std for {key} in {source}")
                    x_slice = x_slice * std + mean
                elif norm_mode is NormalizationMode.MIN_MAX:
                    min_val, max_val = buffer["min"], buffer["max"]
                    assert not torch.isinf(min_val).any(), _no_stats_error_str(f"min for {key} in {source}")
                    assert not torch.isinf(max_val).any(), _no_stats_error_str(f"max for {key} in {source}")
                    x_slice = (x_slice + 1) / 2
                    x_slice = x_slice * (max_val - min_val) + min_val
                else:
                    raise ValueError(norm_mode)
                
                # Update the original tensor
                x[indices] = x_slice

            batch[key] = x  # Update the batch with normalized values

        return batch