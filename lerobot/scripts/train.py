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
import time
import os
from contextlib import nullcontext
from pprint import pformat
from typing import Any, Callable

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
    save_training_step,
    load_training_step,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
    is_launched_with_accelerate,
)
from lerobot.common.constants import (
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
    accelerator= None,
) -> tuple[MetricsTracker, dict]:
    train_loss = 0.0

    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()

    if accelerator:
        with accelerator.accumulate(policy):
            loss, output_dict = policy.forward(batch)
            # TODO(rcadene): policy.unnormalize_outputs(out_dict)

            avg_loss = accelerator.gather(loss.detach().clone().unsqueeze(0)).mean()
            train_loss += avg_loss.item() / accelerator.gradient_accumulation_steps
            
            # NOTE: the operation of unscaling gradients is done inside the backward method
            accelerator.backward(loss)

            ## clip gradients
            if accelerator.sync_gradients:
                trainable_params = list(filter(lambda p: p.requires_grad, policy.parameters()))
                grad_norm = accelerator.clip_grad_norm_(trainable_params, grad_clip_norm)

            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
            
    else:
        with torch.autocast(device_type=device.type) if use_amp and accelerator is None else nullcontext():
            loss, output_dict = policy.forward(batch)
            # TODO(rcadene): policy.unnormalize_outputs(out_dict)
        grad_scaler.scale(loss).backward()
        # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
        grad_scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            grad_clip_norm,
            error_if_nonfinite=False,
        )

        # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        with lock if lock is not None else nullcontext():
            grad_scaler.step(optimizer)
        # Updates the scale for next iteration.
        grad_scaler.update()
        optimizer.zero_grad()

        # Step through pytorch scheduler at every batch instead of epoch
        if lr_scheduler is not None:
            lr_scheduler.step()

    if accelerator and has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
    elif has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item() if accelerator is None else train_loss
    train_metrics.grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    ### modified by Yang Zhang
    # Initialize the accelerator if distributed training is specified

    weight_dtype = torch.float32
    if is_launched_with_accelerate():
        import accelerate
        from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
        accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=cfg.output_dir / cfg.accelerator_logging_dir)
        accelerator = accelerate.Accelerator(
            # mixed_precision will take the os.environ['ACCELERATE_MIXED_PRECISION'] as default value
            # gradient_accumulation_steps will take the os.environ['ACCELERATE_GRADIENT_ACCUMULATION_STEPS'] as default value
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            project_config=accelerator_project_config,
            step_scheduler_with_optimizer=False
        )
        # acquire the mixed precision dtype
        if accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        elif accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        
        cfg.mixed_precision = accelerator.mixed_precision
    else:
        accelerator = None

    if not accelerator or accelerator.is_main_process:
        logging.info(pformat(cfg.to_dict()))

    ##### TODO: (chenyou fan) I'm not sure this works as intended, are the metrics reported correct?
    ##### Currently, we only log stats in the main process
    if accelerator and not accelerator.is_main_process:
        # Disable logging on non-main processes.
        cfg.wandb.enable = False

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True, accelerator=accelerator)
    if accelerator:
        cfg.policy.device = f'cuda:{accelerator.process_index}'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    ### exceptional for Pi0 ###
    ### WARNING: These manual modifications would be deprecated in the future
    ### We would move these DIY configs into the `from_pretrained` method of the Pi0Policy
    if 'bridge' in cfg.dataset.repo_id or 'simplified' in cfg.dataset.repo_id:
        if 'pi0' == cfg.policy.type:
            cfg.policy.tokenizer_max_length = 64

    logging.info("Creating dataset" if not accelerator else "[rank%d] Creating dataset" % accelerator.process_index)

    # with accelerator.main_process_first():
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env" if not accelerator else "[rank%d] Creating env" % accelerator.process_index)
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy" if not accelerator else "[rank%d] Creating policy" % accelerator.process_index)
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        features=getattr(dataset, "intersection_features", None),
    )
    if accelerator:
        policy = policy.to(weight_dtype)
    else:
        policy.to(device)

    if not accelerator or accelerator.is_main_process:
        logging.info(f"policy.config.input_features: {pformat(policy.config.input_features)}")
        logging.info(f"policy.config.output_features: {pformat(policy.config.output_features)}")

    # define save_model hook
    if accelerator:
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    ### option 1
                    # if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    #     model: CogVideoXTransformer3DModel
                    #     model = unwrap_model(accelerator, model)
                    #     model.save_pretrained(
                    #         os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                    #     )
                    # else:
                    #     raise ValueError(f"Unexpected save model: {model.__class__}")
                
                    ### option 2
                    model.save_pretrained(os.path.join(output_dir, PRETRAINED_MODEL_DIR))
                    cfg.save_pretrained(os.path.join(output_dir, PRETRAINED_MODEL_DIR))
                    save_training_step(step, checkpoint_dir)
                    
                    # make sure to pop weight so that corresponding model is not saved again
                    if accelerator.distributed_type != "DEEPSPEED":
                        weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                load_model = model.__class__.from_pretrained(
                    os.path.join(input_dir, PRETRAINED_MODEL_DIR),
                )

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    input_features = policy.config.input_features
    output_features = policy.config.output_features

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume and not accelerator:
        # 根据原生逻辑，在这里会调用resume载入训练所需的stuff
        # NOTE: 这里只适用于单卡训练，非分布式训练
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if not accelerator or accelerator.is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        
        ### TODO: 如果是多任务数据集载入，这里不支持这种episode aware sampler
        if cfg.dataset.repo_id.startswith("["):
            raise ValueError("EpisodeAwareSampler is not supported for multi-task datasets")
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    if accelerator:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)

    if cfg.resume and accelerator:
        accelerator.load_state(cfg.checkpoint_path)
        step = load_training_step(cfg.checkpoint_path)
        if accelerator.is_main_process:
            logging.info(f"Load json from {cfg.checkpoint_path / 'training_step.json'}. Get global step = {step}.")

    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step, accelerator=accelerator,
    )

    if not accelerator or accelerator.is_main_process:
        logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        # print(f"step: {_}")
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        if not accelerator:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
        else:
            for key in batch:
                if key in input_features and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(dtype=weight_dtype, non_blocking=True)
                
                if key in output_features and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(dtype=weight_dtype, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
            accelerator=accelerator,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = (
            cfg.log_freq > 0 and step % cfg.log_freq == 0
            and (not accelerator or accelerator.is_main_process)
        )
        is_saving_step = (
            step % cfg.save_freq == 0 or step == cfg.steps
        )
        is_eval_step = (
            cfg.eval_freq > 0 and step % cfg.eval_freq == 0
            and (not accelerator or accelerator.is_main_process)
        )

        ## add debug when step == 1
        if is_log_step or (step == 1 and (not accelerator or accelerator.is_main_process)):
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            if not accelerator or accelerator.is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
            
            if accelerator:
                accelerator.wait_for_everyone()
                # NOTE: whenever using Accelerate (including the case of DeepSpeed), we should save state on each process
                accelerator.save_state(checkpoint_dir)
            else:
                save_checkpoint(
                    checkpoint_dir,
                    step,
                    cfg,
                    policy,
                    optimizer,
                    lr_scheduler,
                )
            
            if not accelerator or accelerator.is_main_process:
                update_last_checkpoint(checkpoint_dir)
                
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            if accelerator:
                accelerator.wait_for_everyone()

            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp and not accelerator else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy if not accelerator else accelerator.unwrap_model(policy),
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step, accelerator=None,
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()

    if not accelerator or accelerator.is_main_process:
        logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
