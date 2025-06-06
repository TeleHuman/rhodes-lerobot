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
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
    is_launched_with_accelerate,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

from packaging import version

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
    accelerator: Callable = None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp and accelerator is None else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    if accelerator:
        accelerator.backward(loss)
        # REVIEW: It isn't necessary to call it here. It will be called in accelerator.clip_grad_norm_:
        # https://github.com/huggingface/accelerate/blob/main/src/accelerate/accelerator.py#L2628
        # accelerator.unscale_gradients(optimizer=optimizer)
        # if accelerator.is_main_process:
        #     accelerator.wait_for_everyone()
        #     import ipdb; ipdb.set_trace()
        ## option 1
        if accelerator.sync_gradients:
            trainable_params = list(filter(lambda p: p.requires_grad, policy.parameters()))
            grad_norm = accelerator.clip_grad_norm_(trainable_params, grad_clip_norm, error_if_nonfinite=False)
        ## option 2
        # grad_norm = torch.nn.utils.clip_grad_norm_(
        #     policy.parameters(),
        #     grad_clip_norm,
        #     error_if_nonfinite=False,
        # )
        optimizer.step()
    else:
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

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Callable = None):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            # NOTE: When using DeepSpeed, we need save model on each card
            # if accelerator.is_main_process:
            models[0].save_pretrained(os.path.join(output_dir, "transformer_flow"))
            if not cfg.use_deepspeed:
                weights.pop()


        def load_model_hook(models, input_dir):
            from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                load_model = PI0Policy.from_pretrained(
                    input_dir
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    ##### TODO: (chenyou fan) I'm not sure this works as intended, are the metrics reported correct?
    ##### We should probably integrate accelerate's WandBTracker inside our WandBLogger instead.
    
    if accelerator:
        # Disable logging on non-main processes.
        cfg.wandb.enable = False

        if accelerator.is_main_process:
            # TODO: maybe more config parameters for tracking
            log_cfg = {"seed": cfg.seed}
            accelerator.init_trackers("pi0_pretrain", config=log_cfg, init_kwargs={"wandb": {"name": f"pi0_pretrain",
                                                                                    "dir": '/gemini/user/private/wandb'}})

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True, accelerator=accelerator)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    ### exceptional for Pi0 ###
    if 'bridge' in cfg.dataset.repo_id and 'pi0' == cfg.policy.type:
        cfg.policy.tokenizer_max_length = 64

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        features=getattr(dataset, "intersection_features", None),
    )
    policy.to(device)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
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

    # if len(sample_weights) > 0:
    #     sampler = WeightedRandomSampler(
    #         weights=sample_weights,
    #         num_samples=len(dataset),
    #         replacement=True,
    #     )

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
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

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
            and (not accelerator or accelerator.is_main_process)
        )
        is_eval_step = (
            cfg.eval_freq > 0 and step % cfg.eval_freq == 0
            and (not accelerator or accelerator.is_main_process)
        )

        if (is_log_step or step == 1) and (not accelerator or accelerator.is_main_process):
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            elif accelerator:
                # NOTE: wandb_log_dict should be a dictionary-like object containing only types `int`, `float`, or `str`.
                accelerator.log(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            #### TODO: (chenyou fan) 这里看下是不是能换成accelerator的save_state方法
            if accelerator:
                accelerator.wait_for_everyone()
                # NOTE: when using DeepSpeed, we need to save on each card
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
    if is_launched_with_accelerate():
        import accelerate
        from accelerate.utils import ProjectConfiguration
        import os

        # We set step_scheduler_with_optimizer False to prevent accelerate from
        # adjusting the lr_scheduler steps based on the num_processes
        # TODO: i don't know how to get cfg here
        logging_dir = os.path.join(cfg.output_dir, 'logs')
        accelerator_project_config = ProjectConfiguration(
            project_dir=cfg.output_dir, logging_dir=logging_dir)

        accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
            log_with='wandb',
            project_config=accelerator_project_config,
            step_scheduler_with_optimizer=False)
        train(accelerator=accelerator)
    else:
        train()
