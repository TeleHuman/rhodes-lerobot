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
import  tqdm
from math import ceil
from copy import deepcopy
import einops
from datasketches import kll_floats_sketch

from lerobot.common.datasets.utils import load_image_as_numpy

def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.camera_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert c < h and c < w, f"expect channel first images, but instead {batch[key].shape}"

            # sanity check that images are float32 in range [0,1]
            # assert batch[key].dtype == torch.float32, f"expect torch.float32, but instead {batch[key].dtype=}"
            # assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            # assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"

            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        elif batch[key].ndim == 3:  # action
            stats_patterns[key] = "b t c -> c"  # 根据实际情况调整转换规则
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns

def compute_stats(dataset, batch_size=8, num_workers=8, max_num_samples=None, keys_to_use=None):
    if max_num_samples is None:
        max_num_samples = len(dataset)

    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    if keys_to_use is None:
        keys_to_use = list(stats_patterns.keys())

    mean_all = {}
    var_all = {}
    max_d = {}
    min_d = {}

    quantile_keys = ["observation.state", "action"]
    quantile_keys = [k for k in quantile_keys if k in keys_to_use]

    first_item = dataset[0]
    kll_dict = {
        key: [kll_floats_sketch() for _ in range(first_item[key].shape[-1])]
        for key in quantile_keys
    }

    def create_dl(seed):
        gen = torch.Generator().manual_seed(seed)
        return torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, batch_size=batch_size,
            shuffle=True, drop_last=False, generator=gen,
        )

    running_N = 0
    for batch in tqdm.tqdm(create_dl(1337), total=ceil(max_num_samples / batch_size), desc="Pass 1"):
        B = len(batch["index"])
        running_N += B
        for key in keys_to_use:
            if key not in batch:
                continue

            pattern = stats_patterns[key]
            x = batch[key].float()
            bm = einops.reduce(x, pattern, "mean")
            bv = einops.reduce((x - bm) ** 2, pattern, "mean")

            if key not in mean_all:
                mean_all[key] = torch.zeros_like(bm)
                var_all[key] = torch.zeros_like(bv)
                max_d[key] = torch.full_like(bm, -float("inf"))
                min_d[key] = torch.full_like(bm, float("inf"))

            mean_all[key] += B * (bm - mean_all[key]) / running_N
            var_all[key] += B * (bv - var_all[key]) / running_N
            max_d[key] = torch.maximum(max_d[key], einops.reduce(x, pattern, "max"))
            min_d[key] = torch.minimum(min_d[key], einops.reduce(x, pattern, "min"))

            if key in quantile_keys:
                # make sure fractile
                if key == 'action':
                    assert len(x.shape) == 3
                elif key == "observation.state":
                    assert len(x.shape) == 2
                else:
                    assert 0
                # breakpoint()

                arr = x.reshape(-1, x.shape[-1]).cpu().numpy()
                # breakpoint()
                for row in arr:
                    for i, val in enumerate(row):
                        kll_dict[key][i].update(float(val))

        if running_N >= max_num_samples:
            break

    std_all = {k: torch.sqrt(var_all[k]) for k in var_all}

    # 初始化量化统计存储
    quantile_stats = {
        key: {
            "total": 0,
            "mean_q": None,
            "var_q": None,
            "q01": np.array([sk.get_quantile(0.01) for sk in kll_dict[key]]),
            "q99": np.array([sk.get_quantile(0.99) for sk in kll_dict[key]]),
        } for key in quantile_keys
    }

    # 统一 Pass 2 遍历一次 dataset，更新所有 quantile_keys
    for batch in tqdm.tqdm(create_dl(1337), total=ceil(max_num_samples / batch_size), desc="Pass 2"):
        for key in quantile_keys:
            x = batch[key].float()

            if key == 'action':
                assert len(x.shape) == 3
            elif key == "observation.state":
                assert len(x.shape) == 2
            else:
                assert 0
            
            arr = x.reshape(-1, x.shape[-1])
            q01_t = torch.tensor(quantile_stats[key]["q01"], device=arr.device)
            q99_t = torch.tensor(quantile_stats[key]["q99"], device=arr.device)

            mask = ((arr >= q01_t) & (arr <= q99_t)).all(dim=1)
            if mask.sum() == 0:
                continue

            sub = arr[mask]
            B = sub.shape[0]
            mean_b = sub.mean(dim=0)
            var_b = ((sub - mean_b) ** 2).mean(dim=0)

            if quantile_stats[key]["mean_q"] is None:
                quantile_stats[key]["mean_q"] = torch.zeros_like(mean_b)
                quantile_stats[key]["var_q"] = torch.zeros_like(mean_b)

            total = quantile_stats[key]["total"]
            mean_q = quantile_stats[key]["mean_q"]
            var_q = quantile_stats[key]["var_q"]

            mean_q += B * (mean_b - mean_q) / (total + B)
            var_q += B * (var_b - var_q) / (total + B)

            quantile_stats[key]["mean_q"] = mean_q
            quantile_stats[key]["var_q"] = var_q
            quantile_stats[key]["total"] += B

    stats = {}
    for key in keys_to_use:
        if key not in mean_all:
            continue
        stats[key] = {
            "mean_all": mean_all[key],
            "std_all": std_all[key],
            "min": min_d[key],
            "max": max_d[key],
        }
        if key in quantile_keys:
            if quantile_stats[key]["var_q"] is not None:
                stats[key].update({
                    "mean": quantile_stats[key]["mean_q"],
                    "std": torch.sqrt(quantile_stats[key]["var_q"]),
                    "q01": quantile_stats[key]["q01"],
                    "q99": quantile_stats[key]["q99"],
                })

    return stats

def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    """Heuristic to estimate the number of samples based on dataset size.
    The power controls the sample growth relative to dataset size.
    Lower the power for less number of samples.

    For default arguments, we have:
    - from 1 to ~500, num_samples=100
    - at 1000, num_samples=177
    - at 2000, num_samples=299
    - at 5000, num_samples=594
    - at 10000, num_samples=1000
    - at 20000, num_samples=1681
    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:
        # no downsampling needed
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    sampled_indices = sample_indices(len(image_paths))

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        # we load as uint8 to reduce memory usage
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)  # data is a list of image paths
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # finally, we normalize and remove batch dim for images
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats


def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
    for i in range(len(stats_list)):
        for fkey in stats_list[i]:
            for k, v in stats_list[i][fkey].items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                    )
                if v.ndim == 0:
                    raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
                if k == "count" and v.shape != (1,):
                    raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
                # bypass depth check
                if "image" in fkey and k != "count":
                    if "depth" not in fkey and v.shape != (3, 1, 1):
                        raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")
                    if "depth" in fkey and v.shape != (1, 1, 1):
                        raise ValueError(f"Shape of '{k}' must be (1,1,1), but is {v.shape} instead.")



def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregates stats for a single feature."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    # Prepare weighted mean by matching number of dimensions
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # Compute the weighted mean
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # Compute the variance using the parallel algorithm
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats
