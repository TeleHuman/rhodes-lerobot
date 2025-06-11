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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange
from collections.abc import MutableMapping
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import add_envs_task, check_env_attributes_and_types, preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.random_utils import set_seed
from collections import OrderedDict
from transforms3d.euler import euler2axangle
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig


##################the functions I defined###################

def pretty_print_observation(obs: dict, indent: int = 0, key_name: str = "obs"):
    prefix = " " * indent

    for k, v in obs.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}/")
            pretty_print_observation(v, indent + 2, key_name=k)
        elif hasattr(v, "shape"):
            shape = tuple(v.shape)
            shape_str = str(shape)  
            dt = getattr(v, "dtype", None)
            dtype_str = dt.name if hasattr(dt, "name") else type(v).__name__
            print(f"{prefix}{k:<20} shape={shape_str:<15} dtype={dtype_str}")
        else:
            s = str(v)
            if len(s) > 80:
                s = s[:77] + "..."
            print(f"{prefix}{k:<20} type={type(v).__name__:<10} value={s}")


def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flattens a nested dictionary, concatenating keys with a separator.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



def qpos_to_xyzrpy_pad_gripper(agent_qpos: np.ndarray) -> np.ndarray:
    """
    将 agent_qpos (B,8) -> (B,8)，
    输入格式 assumed: [x,y,z, qx,qy,qz,qw, gripper]
    输出格式         : [x,y,z, roll,pitch,yaw, pad, gripper]
    pad 列全 0。
    """
    pos     = agent_qpos[:, 0:3]    # (B,3)
    quat    = agent_qpos[:, 3:7]    # (B,4)  [qx, qy, qz, qw]
    gripper = agent_qpos[:, 7:8]    # (B,1)
    x = quat[:, 0]; y = quat[:, 1]; z = quat[:, 2]; w = quat[:, 3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # clamp
    pitch = np.where(
        np.abs(sinp) >= 1.0,
        np.sign(sinp) * (np.pi / 2),
        np.arcsin(sinp),
    )
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    B = agent_qpos.shape[0]
    pad = np.zeros((B, 1), dtype=agent_qpos.dtype)
    euler = np.stack([roll, pitch, yaw], axis=1)  # (B,3)
    out = np.concatenate([pos, euler, pad, gripper], axis=1)
    return out


def myprocess_observation(observation):
    """
    Processes the nested observation dictionary to extract relevant features
    and return them in a format suitable for the policy.
    """
    # Flatten the nested dictionary
    flat_obs = flatten_dict(observation)

    # Extract agent_qpos as action (agent_pos)
    agent_pos = flat_obs.get('agent_qpos', None)   #8
    tcp_pos= flat_obs.get('extra_tcp_pose', None)   #7

    # Extract image data (overhead_camera/rgb)
    overhead_camera_rgb = flat_obs.get('image_3rd_view_camera_rgb', None)
    base_camera_rgb = flat_obs.get('image_base_camera_rgb', None)
    if agent_pos is not None:
        agent_pos = np.array(agent_pos)  # Convert to numpy array if needed
    if tcp_pos is not None:
        tcp_pos = np.array(tcp_pos)  # Convert to numpy array if needed
        # agent_pos=np.concatenate([agent_pos,tcp_pos],axis=-1)
    agent_pos=qpos_to_xyzrpy_pad_gripper(agent_pos)
    return {
        "agent_pos": agent_pos,
        "pixels":{
        "overhead": overhead_camera_rgb,
        "base":    base_camera_rgb,
            }
         #'action':actions
    }



def to_native(x):
   
    if isinstance(x, dict):
        return {k: to_native(v) for k,v in x.items()}
    if isinstance(x, list):
        return [to_native(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def pi0_raw_action_2_action(
    raw_actions: np.ndarray,
    policy_setup: str = "widowx_bridge",
    previous_gripper_action=None,
    sticky_action_is_on=None,
    gripper_action_repeat=None,
    sticky_gripper_action=None,
):
    """
    这个函数参照Simpler_env中对于动作的处理，将策略输出的动作处理后输入环境
    raw_actions: (B, 7+)，这里假设前 7 维是 [x,y,z, rx,ry,rz, open_gripper]
    返回:
      actions:             np.ndarray, shape (B,7)  -> [x,y,z, ax,ay,az, grip]
      previous_gripper_action:  np.ndarray, shape (B,1)
      sticky_action_is_on:      np.ndarray, shape (B,) bool
      gripper_action_repeat:    np.ndarray, shape (B,) int
      sticky_gripper_action:    np.ndarray, shape (B,1)
    """
    B = raw_actions.shape[0]

    # 状态初始化（仅第一次）
    if previous_gripper_action is None:
        previous_gripper_action = np.zeros((B, 1))
    if sticky_action_is_on is None:
        sticky_action_is_on = np.zeros(B, dtype=bool)
    if gripper_action_repeat is None:
        gripper_action_repeat = np.zeros(B, dtype=int)
    if sticky_gripper_action is None:
        sticky_gripper_action = np.zeros((B, 1))

    # 拆分动作向量
    world_vectors   = raw_actions[:,  :3]   # (B,3)
    rotation_deltas = raw_actions[:, 3:6]   # (B,3)
    open_grippers   = raw_actions[:, 6:7]   # (B,1)

    # Euler -> axis-angle（loop）
    axangles = []
    for r, p, y in rotation_deltas:
        ax, ang = euler2axangle(r, p, y)
        axangles.append(ax * ang)
    rot_axangles = np.stack(axangles, axis=0)  # (B,3)

    # 计算初始相对 gripper diff
    relative_gripper_action = previous_gripper_action - open_grippers  # (B,1)

    # 根据不同后端循环更新 gripper 状态
    if policy_setup == "google_robot":
        sticky_gripper_num_repeat = 10  # 或从 cfg 里读

        # 对每条样本做 sticky 逻辑
        for i in range(B):
            # 检测“明显翻转”触发 sticky
            if abs(relative_gripper_action[i, 0]) > 0.5 and not sticky_action_is_on[i]:
                sticky_action_is_on[i]       = True
                sticky_gripper_action[i, 0]  = relative_gripper_action[i, 0]
                previous_gripper_action[i, 0]= open_grippers[i, 0]

            # 如果正在 sticky，就重复同一个动作，并计数
            if sticky_action_is_on[i]:
                gripper_action_repeat[i] += 1
                relative_gripper_action[i, 0] = sticky_gripper_action[i, 0]

                # 若达到重复上限，则重置 sticky
                if gripper_action_repeat[i] >= sticky_gripper_num_repeat:
                    sticky_action_is_on[i]       = False
                    gripper_action_repeat[i]     = 0
                    sticky_gripper_action[i, 0]  = 0.0

            else:
                # 非 sticky 状态，正常差值 already in relative_gripper_action
                previous_gripper_action[i, 0] = open_grippers[i, 0]

        gripper_out = relative_gripper_action  # (B,1)

    elif policy_setup == "widowx_bridge":
        # 直接二值化
        gripper_out = 2.0 * (open_grippers > 0.5) - 1.0  # (B,1)

    else:
        raise ValueError(f"Unknown policy_setup: {policy_setup}")

    # 最终拼接：(B,3) + (B,3) + (B,1) → (B,7)
    actions = np.concatenate([world_vectors, rot_axangles, gripper_out], axis=1)

    return actions


##################the functions I defined###################

def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()

    observation, info = env.reset(seed=seeds)
    #print("the observation before my conversion to it looks like   →")
    #pretty_print_observation(observation)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    
    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done):
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        #import ipdb ;ipdb.set_trace()
        observation= myprocess_observation(observation)
        
        observation = preprocess_observation(observation)       
        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)
     
        with torch.inference_mode():
            action = policy.select_action(observation)
        
        # Convert to CPU / numpy.
        action = action.to("cpu").squeeze(0).numpy()
        if isinstance(env,gym.vector.VectorEnv):
            assert action.ndim == 2,    "Action dimensions should be (batch, action_dim)"
        else:
           pass
        
        # Apply the next action.
        action=pi0_raw_action_2_action(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if render_callback is not None:
            render_callback(env)
        
        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available of none of the envs finished.
        if "final_info" in info:
            successes = [info["success"] if info is not None else False for info in info["final_info"]]
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        done = terminated | truncated | done

        all_actions.append(torch.from_numpy(action))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))
        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()
    
  
    all_final_info=info["final_info"]
    # Track the final observation.
    if return_observations:
        observation= myprocess_observation(observation)
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
        "all_final_info":all_final_info,
    }
   
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret["observation"] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None
    all_final_info=[]
    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env,
            policy,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        
        ori_final_info_this_batch=rollout_data["all_final_info"].tolist()
        
        log_final_info_this_batch=[]
        for entry in ori_final_info_this_batch :
            if entry is None:
                ori_final_info_this_batch.append(None)
                continue
            stats = entry.get("episode_stats")
            if isinstance(stats, OrderedDict):
                entry["episode_stats"] = dict(stats)
            log_final_info_this_batch.append(entry)
            
        log_final_info_this_batch= to_native(log_final_info_this_batch)
        all_final_info.extend(log_final_info_this_batch)
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                       #env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
                "final_info":all_final_info,
            }
            for i, (sum_reward, max_reward, success, seed,all_final_info) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    all_final_info[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data["action"].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            "action": rollout_data["action"][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            "next.done": rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            "next.reward": rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data["observation"]:
            ep_dict[key] = rollout_data["observation"][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    #logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy(
            env,
            policy,
            cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
        )
    print(info["aggregated"])

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    env.close()

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
