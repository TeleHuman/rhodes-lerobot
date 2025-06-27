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
import logging
from pprint import pformat
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import WeightedRandomSampler

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]

        # compatible with RoboMind-like dataset
        if key.startswith("actions.") and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    # image_transforms = (
    #     ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    # )
    
    if cfg.dataset.repo_id.startswith("["):
        cfg.dataset.repo_id = eval(cfg.dataset.repo_id)

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        # 获取observation.images开头的第一个key
        image_key = next(
            (key for key in ds_meta.features.keys() if key.startswith("observation.images.")),
            None
        )
        if image_key is None:
            raise ValueError("No image key found in the dataset")
        
        # 获取图像维度名称列表
        image_dim_names = ds_meta.features[image_key]['names']
        
        # 找到height和width在names中的索引位置
        height_idx = image_dim_names.index('height') if 'height' in image_dim_names else None
        width_idx = image_dim_names.index('width') if 'width' in image_dim_names else None
        
        if height_idx is None or width_idx is None:
            raise ValueError("Could not find 'height' or 'width' in image dimension names")
            
        # 根据索引从shape中获取实际的高度和宽度值
        image_shape = ds_meta.features[image_key]['shape']
        height = image_shape[height_idx]
        width = image_shape[width_idx]

        ### Option 1: use piohfive sequential transform
        # image_transforms = ImageTransforms.create_piohfive_sequential_transform(
        #     (height, width)
        # ) if cfg.dataset.image_transforms.enable else None
        ### Option 2: use original transform in Lerobot
        image_transforms = ImageTransforms(cfg.dataset.image_transforms, height=height, width=width) if cfg.dataset.image_transforms.enable else None

        wrist_transforms = ImageTransforms(cfg.dataset.wrist_transforms, height=None, width=None) if cfg.dataset.wrist_transforms.enable else None

        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            wrist_transforms=wrist_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
    elif isinstance(cfg.dataset.repo_id, list):
        delta_timestamps_ds_dict = {}
        for repo_id in cfg.dataset.repo_id:
            ds_meta = LeRobotDatasetMetadata(
                repo_id, root=Path(cfg.dataset.root) / repo_id, revision=cfg.dataset.revision
            )
            delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
            delta_timestamps_ds_dict[repo_id] = delta_timestamps

        # import ipdb; ipdb.set_trace()
        ### HACK: to be considered more
        # sampler = WeightedRandomSampler(
        #     weights=sample_weights,
        #     num_samples=32,
        #     replacement=True,
        # )
        
        ### go!go!go!出发咯
        #### 以这个为主，concatdataset放弃
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            delta_timestamps_ds_dict=delta_timestamps_ds_dict,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
            policy_normalization_mapping=cfg.policy.normalization_mapping,
            img_resize_shape=cfg.policy.resize_imgs_with_padding,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: \n"
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    else:
        raise ValueError(f"Invalid dataset repo_id: {cfg.dataset.repo_id}")

    if cfg.dataset.use_imagenet_stats:
        if isinstance(dataset, MultiLeRobotDataset):
            for ds in dataset._datasets:
                for key in ds.meta.camera_keys:
                    for stats_type, stats in IMAGENET_STATS.items():
                        ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        else:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
