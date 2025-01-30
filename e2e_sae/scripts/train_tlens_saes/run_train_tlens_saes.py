"""Script for training SAEs on top of a transformerlens model.

Usage:
    python run_train_tlens_saes.py <path/to/config.yaml>
"""
import math
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
from typing import Optional, cast

import fire
import torch
from datasets import IterableDataset
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from e2e_sae.data import create_data_loader
from e2e_sae.hooks import SAEActs
from e2e_sae.loader import load_pretrained_saes, load_tlens_model
from e2e_sae.log import logger
from e2e_sae.losses import calc_loss
from e2e_sae.metrics import (
    ActFrequencyMetrics,
    calc_output_metrics,
    calc_sparsity_metrics,
    collect_act_frequency_metrics,
)
from e2e_sae.models.transformers import SAETransformer
from e2e_sae.parallel_wandb import WandbWrapper, create_wandb_procs_queues_wrappers
from e2e_sae.scripts.train_tlens_saes.tlens_sae_train_config import Config, get_run_name
from e2e_sae.types import Samples
from e2e_sae.utils import (
    filter_names,
    get_cosine_schedule_with_warmup,
    get_linear_lr_schedule,
    load_config,
    replace_pydantic_model,
    save_module,
    set_seed, GPUMemTracker,
)


@torch.inference_mode()
def evaluate(
    config: Config,
    model: SAETransformer,
    device: torch.device,
    cache_positions: list[str] | None,
    log_resid_reconstruction: bool = True,
    sae_variant_idx: int = 0
) -> dict[str, float]:
    """Evaluate the model on the eval dataset.

    Accumulates metrics over the entire eval dataset and then divides by the total number of tokens.

    Args:
        config: The config object.
        model: The SAETransformer model.
        device: The device to run the model on.
        cache_positions: The positions to cache activations at.
        log_resid_reconstruction: Whether to log the reconstruction loss and explained variance
            at hook_resid_post in all layers.
        sae_variant_idx: which SAE variant (that's currently loaded in the model) to evaluate
    Returns:
        Dictionary of metrics.
    """
    model.saes.eval()

    vram_tracker = GPUMemTracker()

    eval_config = config
    eval_cache_positions = cache_positions
    eval_loss_config_updates = {}
    if log_resid_reconstruction and config.loss.in_to_orig is None:
        # Update cache_positions with all hook_resid_post positions
        all_resids = [f"blocks.{i}.hook_resid_post" for i in range(model.tlens_model.cfg.n_layers)]
        # Record the reconstruction loss and explained var at hook_resid_post by setting
        # in_to_orig.total_coeff to 0.0
        eval_loss_config_updates.update(
            {"in_to_orig": {"hook_positions": all_resids, "total_coeff": 0.0}}
        )
        eval_cache_positions = list(
            set(all_resids) | (set(cache_positions) if cache_positions else set())
        )
    if config.loss.logits_kl is None:
        # If we're not training with logits_kl ensure that we eval with it
        eval_loss_config_updates.update({"logits_kl": {"coeff": 0.0}})

    # Use a different seed for evaluation than for training if eval seed not explicitly set
    eval_config = replace_pydantic_model(
        config,
        {"loss": eval_loss_config_updates, "seed": config.seed + 42},
    )

    vram_tracker.check(f"before create data loader for evaluate() on SAE variant {sae_variant_idx}")
    assert eval_config.eval_data is not None, "No eval dataset specified in the config."
    eval_loader = create_data_loader(
        eval_config.eval_data, batch_size=eval_config.batch_size, global_seed=eval_config.seed
    )[0]

    if eval_config.eval_n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = None if isinstance(eval_loader.dataset, IterableDataset) else len(eval_loader)
    else:
        n_batches = math.ceil(eval_config.eval_n_samples / eval_config.batch_size)

    total_tokens = 0
    # Accumulate metrics over the entire eval dataset and later divide by the total number of tokens
    metrics: dict[str, float] = {}

    vram_tracker.check(f"before start loop through batches of eval data for SAE variant {sae_variant_idx}")
    for batch_idx, batch in tqdm(enumerate(eval_loader), total=n_batches, desc="Eval Steps"):
        if n_batches is not None and batch_idx >= n_batches:
            break

        vram_tracker.check(f"before move {batch_idx}th batch of eval data for SAE variant {sae_variant_idx} "
                           f"to device")
        tokens = batch[eval_config.eval_data.column_name].to(device=device)
        n_tokens = tokens.shape[0] * tokens.shape[1]
        total_tokens += n_tokens
        vram_tracker.check(f"before model.forward_raw() on {batch_idx}th batch of eval data for SAE variant "
                           f"{sae_variant_idx}")

        # Run through the raw transformer without SAEs
        orig_logits, orig_acts = model.forward_raw(
            tokens=tokens,
            run_entire_model=True,
            final_layer=None,
            cache_positions=eval_cache_positions,
        )
        vram_tracker.check(f"before model.forward() on {batch_idx}th batch of eval data for SAE variant "
                           f"{sae_variant_idx}")
        # Run through the SAE-augmented model
        new_logits, new_acts = model.forward(
            tokens=tokens,
            sae_positions=model.raw_sae_positions,
            cache_positions=eval_cache_positions,
            orig_acts=orig_acts,  # more efficient to skip the computations for layers earlier than the first SAE pos
            sae_variant_idx=sae_variant_idx,
            should_run_to_logits=True
        )
        assert new_logits is not None, "new_logits should not be None during evaluation."
        vram_tracker.check(f"before calc_loss() on {batch_idx}th batch of eval data for SAE variant "
                           f"{sae_variant_idx}")

        raw_batch_loss_dict = calc_loss(
            orig_acts=orig_acts,
            new_acts=new_acts,
            orig_logits=orig_logits,
            new_logits=new_logits,
            loss_configs=eval_config.loss,
            sae_variant_idx=sae_variant_idx,
            is_log_step=True,
            train=False,
        )[1]
        vram_tracker.check(f"before calc output and sparsity metrics on {batch_idx}th batch of eval data for "
                           f"SAE variant {sae_variant_idx}")
        batch_loss_dict = {k: v.item() for k, v in raw_batch_loss_dict.items()}
        batch_output_metrics = calc_output_metrics(
            tokens=tokens, orig_logits=orig_logits, new_logits=new_logits, train=False
        )

        # TODO in matryoshka case, add logic to compute eval metrics for the sub-dictionaries

        sparsity_metrics = calc_sparsity_metrics(new_acts=new_acts, train=False)
        vram_tracker.check(f"before update metrics dict based on {batch_idx}th batch of eval data for "
                           f"SAE variant {sae_variant_idx}")

        # Update the global metric dictionary
        for k, v in {**batch_loss_dict, **batch_output_metrics, **sparsity_metrics}.items():
            metrics[k] = metrics.get(k, 0.0) + v * n_tokens

    # Get the mean for all metrics
    for key in metrics:
        metrics[key] /= total_tokens

    model.saes.train()
    return metrics


@logging_redirect_tqdm()
def train(
    config: Config,
    model: SAETransformer,
    train_loader: DataLoader[Samples],
    trainable_param_names: list[str],
    device: torch.device,
    run_names: list[str],
    wandb_wrapper: Optional[WandbWrapper],
    cache_positions: list[str] | None = None,
) -> None:
    model.saes.train()

    vram_tracker = GPUMemTracker()

    is_local = config.loss.logits_kl is None and cache_positions is None

    vram_tracker.check("b4 specify trainable parameters and create optimizer in train()")
    for name, param in model.named_parameters():
        if name.startswith("saes.") and name.split("saes.")[1] in trainable_param_names:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
    )

    effective_batch_size = config.effective_batch_size or config.batch_size
    n_gradient_accumulation_steps = effective_batch_size // config.batch_size

    if config.lr_schedule == "cosine":
        assert config.n_samples is not None, "Cosine schedule requires n_samples."
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_samples // effective_batch_size,
            num_training_steps=config.n_samples // effective_batch_size,
            min_lr_factor=config.min_lr_factor,
        )
    else:
        assert config.lr_schedule == "linear"
        lr_schedule = get_linear_lr_schedule(
            warmup_samples=config.warmup_samples,
            cooldown_samples=config.cooldown_samples,
            n_samples=config.n_samples,
            effective_batch_size=effective_batch_size,
            min_lr_factor=config.min_lr_factor,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    if config.n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = None if isinstance(train_loader.dataset, IterableDataset) else len(train_loader)
    else:
        n_batches = math.ceil(config.n_samples / config.batch_size)

    final_layer = None
    if all(name.startswith("blocks.") for name in model.raw_sae_positions) and is_local:
        # We don't need to run through the whole model for local runs
        final_layer = max([int(name.split(".")[1]) for name in model.raw_sae_positions]) + 1



    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dirs = [config.save_dir / f"{run_name}_{timestamp}" if config.save_dir else None for run_name in run_names]

    total_samples = 0
    total_samples_at_last_save = 0
    total_samples_at_last_eval = 0
    total_tokens = 0
    grad_updates = 0
    grad_norm: float | None = None
    samples_since_act_frequency_collection: int = 0
    act_frequency_metrics_trackers: list[ActFrequencyMetrics | None] = [None] * len(run_names)

    def save_checkpoint_of_sae_variant(total_samples_so_far: int, sae_variant_idx: int):
        saes_backup_file_name = f"samples_{total_samples_so_far}.pt"
        curr_save_dir = save_dirs[sae_variant_idx]
        save_module(
            config_dict=config.model_dump(mode="json"),
            save_dir=curr_save_dir,
            module=torch.nn.ModuleDict({sae_key: sae for sae_key, sae in model.saes.items()
                                        if sae_key.endswith(f"-{sae_variant_idx}")}),
            model_filename=saes_backup_file_name,
            config_filename="final_config.yaml",
        )
        if wandb_wrapper:
            wandb_wrapper.save(sae_variant_idx, str(curr_save_dir / saes_backup_file_name), policy="now",
                               base_path=curr_save_dir)

    for batch_idx, batch in tqdm(enumerate(train_loader), total=n_batches, desc="Steps"):
        vram_tracker.check(f"before fetching {batch_idx}th batch of tokens and moving them to device")
        tokens: Int[Tensor, "batch pos"] = batch[config.train_data.column_name].to(device=device)

        total_samples += tokens.shape[0]
        total_tokens += tokens.shape[0] * tokens.shape[1]
        samples_since_act_frequency_collection += tokens.shape[0]

        # Note that is_last_batch will always be False for iterable datasets with n_samples=None. In
        # that case, we will never know when the final batch is reached.
        is_last_batch: bool = n_batches is not None and batch_idx == n_batches - 1
        is_grad_step: bool = (batch_idx + 1) % n_gradient_accumulation_steps == 0
        is_eval_step: bool = config.eval_every_n_samples is not None and (
            (batch_idx == 0)
            or total_samples - total_samples_at_last_eval >= config.eval_every_n_samples
            or is_last_batch
        )
        is_collect_act_frequency_step: bool = config.collect_act_frequency_every_n_samples > 0 and (
            batch_idx == 0
            or (
                samples_since_act_frequency_collection
                >= config.collect_act_frequency_every_n_samples
            )
        )
        is_log_step: bool = (
            batch_idx == 0
            or (is_grad_step and (grad_updates + 1) % config.log_every_n_grad_steps == 0)
            or is_eval_step
            or is_last_batch
        )
        is_save_model_step: bool = save_dirs[0] is not None and (
            (
                config.save_every_n_samples
                and total_samples - total_samples_at_last_save >= config.save_every_n_samples
            )
            or is_last_batch
        )
        logger.info(f"starting {batch_idx}'th batch; is grad update step={is_grad_step}; is eval step={is_eval_step};"
                    f"is collect_act_freq step={is_collect_act_frequency_step}; is log step={is_log_step};"
                    f"is save model step={is_save_model_step}")

        vram_tracker.check(f"before doing forward_raw() for {batch_idx}th batch of tokens")
        # Run through the raw transformer without SAEs
        orig_logits, orig_acts = model.forward_raw(
            tokens=tokens,
            run_entire_model=not is_local,
            final_layer=final_layer,
            cache_positions=cache_positions,
        )
        safe_orig_logits = orig_logits.detach().clone()

        for sae_spec_idx, sae_spec in enumerate(model.sae_specs):
            assert len(model.raw_sae_positions) == 1 or not sae_spec.is_matryoshka, \
                ("code for Matryoshka training of SAE's with e2e and/or downstream-recon loss does not currently "
                 "support multiple SAE positions in the model")
            assert not is_local or not sae_spec.is_matryoshka, \
                "this codebase's support for Matryoshka training of SAE's doesn't currently include myopic training"

            vram_tracker.check(f"before running model.forward() for sae variant {sae_spec_idx} on "
                               f"{batch_idx}th batch of tokens")
            # Run through the SAE-augmented model
            new_logits, new_acts = model.forward(
                tokens=tokens,
                sae_positions=model.raw_sae_positions,
                cache_positions=cache_positions,
                orig_acts=orig_acts,
                sae_variant_idx=sae_spec_idx,
                should_run_to_logits=not is_local
            )

            vram_tracker.check(f"before calculating losses for sae variant {sae_spec_idx} on "
                               f"{batch_idx}th batch of tokens")
            loss, loss_dict = calc_loss(
                orig_acts=orig_acts,
                new_acts=new_acts,
                orig_logits=None if new_logits is None else safe_orig_logits,
                new_logits=new_logits,
                loss_configs=config.loss,
                sae_variant_idx=sae_spec_idx,
                is_log_step=is_log_step,
            )

            num_reconstructions_for_loss_calcs = ((len(sae_spec.matryoshka_group_proportions) + 1)
                                                  if sae_spec.is_matryoshka else 1)
            overall_loss_divisor = n_gradient_accumulation_steps * num_reconstructions_for_loss_calcs
            overall_loss = loss / overall_loss_divisor

            if sae_spec.is_matryoshka:
                sae_raw_pos = model.raw_sae_positions[0]
                curr_sae_cached_acts_key = SAETransformer.sae_raw_pos_to_cached_acts_key(sae_raw_pos, sae_spec_idx)
                sae_cached_acts = new_acts[curr_sae_cached_acts_key]
                assert isinstance(sae_cached_acts, SAEActs)
                intermediate_recons = sae_cached_acts.tertiary_SAE_results.intermediate_reconstructions

                for intermed_recon_idx, intermediate_reconstruct in enumerate(intermediate_recons):
                    vram_tracker.check(f"before model.forward() for {intermed_recon_idx}th intermediate"
                                       f"reconstruction for (matryoshka) sae variant {sae_spec_idx} on "
                                       f"{batch_idx}th batch of tokens")
                    logits_w_curr_intermed_recon, acts_w_curr_intermed_recon = model.forward(
                        tokens=tokens, sae_positions=[], cache_positions=cache_positions,
                        orig_acts=orig_acts, sae_variant_idx=sae_spec_idx, should_run_to_logits=not is_local,
                        inject_positions_activations={sae_raw_pos: intermediate_reconstruct}
                    )

                    vram_tracker.check(f"before loss calculation for {intermed_recon_idx}th intermediate"
                                       f"reconstruction for (matryoshka) sae variant {sae_spec_idx} on "
                                       f"{batch_idx}th batch of tokens")
                    loss_w_curr_intermed_recon, loss_dict_w_curr_intermed_recon = calc_loss(
                        orig_acts=orig_acts, new_acts=acts_w_curr_intermed_recon,
                        orig_logits=None if logits_w_curr_intermed_recon is None else safe_orig_logits,
                        new_logits=logits_w_curr_intermed_recon, loss_configs=config.loss,
                        sae_variant_idx=sae_spec_idx, is_log_step=is_log_step
                    )

                    loss_w_curr_intermed_recon = loss_w_curr_intermed_recon / overall_loss_divisor
                    overall_loss += loss_w_curr_intermed_recon

                    loss_dict.update({f"{k}/matryoshka{sae_spec.matryoshka_group_proportions[intermed_recon_idx]}": v
                                      for k, v in loss_dict_w_curr_intermed_recon.items()})

            vram_tracker.check(f"before loss.backward() for sae variant {sae_spec_idx} on "
                               f"{batch_idx}th batch of tokens")
            overall_loss.backward()
            overall_loss_val = overall_loss.item()
            vram_tracker.check(f"after loss.backward() for sae variant {sae_spec_idx} on "
                               f"{batch_idx}th batch of tokens")

            if is_grad_step:
                if config.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.saes.parameters(), config.max_grad_norm
                    ).item()
                for raw_sae_pos in model.raw_sae_positions:
                    curr_sae = model.saes[SAETransformer.sae_raw_pos_to_sae_key(raw_sae_pos, sae_spec_idx)]
                    if hasattr(curr_sae, 'make_decoder_weights_and_grad_unit_norm'):
                        curr_sae.make_decoder_weights_and_grad_unit_norm()
                optimizer.step()
                optimizer.zero_grad()
                grad_updates += 1
                scheduler.step()
                vram_tracker.check(f"after {grad_updates}th (1-based) grad update for "
                                   f"sae variant {sae_spec_idx} (on {batch_idx}th batch of tokens)")

            if is_collect_act_frequency_step and act_frequency_metrics_trackers[sae_spec_idx] is None:
                # Start collecting activation frequency metrics for next config.act_frequency_n_tokens
                act_frequency_metrics_trackers[sae_spec_idx] = ActFrequencyMetrics(
                    dict_sizes={
                        hook_pos: new_act_pos.c.shape[-1]
                        for hook_pos, new_act_pos in new_acts.items()
                        if isinstance(new_act_pos, SAEActs)
                    },
                    device=device,
                )
                samples_since_act_frequency_collection = 0

            if act_frequency_metrics_trackers[sae_spec_idx] is not None:
                act_frequency_metrics_trackers[sae_spec_idx].update_dict_el_frequencies(
                    new_acts, batch_tokens=tokens.shape[0] * tokens.shape[1]
                )
                if act_frequency_metrics_trackers[sae_spec_idx].tokens_used >= config.act_frequency_n_tokens:
                    # TODO this might be a good spot to handle resampling of stubbornly dead latents
                    # Finished collecting activation frequency metrics
                    metrics = act_frequency_metrics_trackers[sae_spec_idx].collect_for_logging(
                        log_wandb_histogram=config.wandb_project is not None
                    )
                    metrics["total_tokens"] = total_tokens
                    if wandb_wrapper:
                        # TODO: Log when not using wandb too
                        wandb_wrapper.log(sae_spec_idx, metrics, step=total_samples)
                    act_frequency_metrics_trackers[sae_spec_idx] = None
                    samples_since_act_frequency_collection = 0

            if is_log_step:
                tqdm.write(
                    f"Samples {total_samples} Batch_idx {batch_idx} GradUpdates {grad_updates} "
                    f"Loss {overall_loss_val:.5f}"
                )
                if wandb_wrapper:
                    log_info = {
                        "loss": overall_loss_val,
                        "grad_updates": grad_updates,
                        "total_tokens": total_tokens,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    log_info.update({k: v.item() for k, v in loss_dict.items()})
                    if grad_norm is not None:
                        log_info["grad_norm"] = grad_norm  # Norm of grad before clipping

                    sparsity_metrics = calc_sparsity_metrics(new_acts=new_acts)
                    log_info.update(sparsity_metrics)

                    if new_logits is not None:
                        train_output_metrics = calc_output_metrics(
                            tokens=tokens,
                            orig_logits=orig_logits.detach().clone(),
                            new_logits=new_logits.detach().clone(),
                        )
                        log_info.update(train_output_metrics)

                    if is_eval_step:
                        # TODO investigate whether/how this evaluate step could be broken out of the for loop over SAE
                        #  variants, so then evaluate() could share orig_acts (for a given batch of eval data) between
                        #  the evaluations of the different SAE variants
                        eval_metrics = evaluate(
                            config=config, model=model, device=device, cache_positions=cache_positions,
                            sae_variant_idx=sae_spec_idx
                        )
                        total_samples_at_last_eval = total_samples
                        log_info.update(eval_metrics)

                    wandb_wrapper.log(sae_spec_idx, log_info, step=total_samples)

            if is_save_model_step:
                assert save_dirs[sae_spec_idx] is not None
                total_samples_at_last_save = total_samples
                save_checkpoint_of_sae_variant(total_samples, sae_spec_idx)

        if is_last_batch:
            break

    # If the model wasn't saved at the last step of training (which may happen if n_samples: null
    # and the dataset is an IterableDataset), save it now.
    if save_dirs[0] and not (save_dirs[0] / f"samples_{total_samples}_sae_variant_0.pt").exists():
        for sae_spec_idx in range(len(model.sae_specs)):
            save_checkpoint_of_sae_variant(total_samples, sae_spec_idx)

    if wandb_wrapper:
        for sae_spec_idx in range(len(model.sae_specs)):
            # Collect and log final activation frequency metrics
            metrics = collect_act_frequency_metrics(
                model=model,
                data_config=config.train_data,
                batch_size=config.batch_size // 2,  # Hack to prevent OOM. TODO: Solve this properly
                global_seed=config.seed,
                device=device,
                n_tokens=config.act_frequency_n_tokens,
                sae_variant_idx=sae_spec_idx
            )
            wandb_wrapper.log(sae_spec_idx, metrics)


def main(
    config_path_or_obj: Path | str | Config  # , sweep_config_path: Path | str | None = None
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path_or_obj, config_model=Config)

    vram_tracker = GPUMemTracker.initialize(device, 0.05)

    base_run_name = get_run_name(config)
    run_name_suffixes = [sae_spec.to_run_name_suffix() for sae_spec in config.saes.sae_specs]
    run_names = [base_run_name + run_name_suffix for run_name_suffix in run_name_suffixes]

    wandb_log_queues: list[mp.Queue] = []
    wandb_procs: list[mp.Process] = []
    wandb_wrapper: Optional[WandbWrapper] = None

    if config.wandb_project:
        wandb_log_queues, wandb_procs, wandb_wrapper = create_wandb_procs_queues_wrappers(
            config, base_run_name, run_name_suffixes)

    set_seed(config.seed)
    logger.info(config)

    vram_tracker.check("b4 train loader")
    train_loader = create_data_loader(
        config.train_data, batch_size=config.batch_size, global_seed=config.seed
    )[0]
    vram_tracker.check("b4 load TLens model")
    tlens_model = load_tlens_model(
        tlens_model_name=config.tlens_model_name, tlens_model_path=config.tlens_model_path,
        tlens_model_dtype=config.tlens_model_dtype
    )

    raw_sae_positions = filter_names(list(tlens_model.hook_dict.keys()), config.saes.sae_positions)
    cache_positions: list[str] | None = None
    if config.loss.in_to_orig is not None:
        assert set(config.loss.in_to_orig.hook_positions).issubset(
            set(tlens_model.hook_dict.keys())
        ), "Some hook_positions in config.loss.in_to_orig.hook_positions are not in the model."
        # Don't add a cache position if there is already an SAE at that position which will cache
        # the inputs anyway
        cache_positions = [
            pos for pos in config.loss.in_to_orig.hook_positions if pos not in raw_sae_positions
        ]

    vram_tracker.check("b4 create SAETransformer")
    model = SAETransformer(
        tlens_model=tlens_model,
        raw_sae_positions=raw_sae_positions,
        saes_config=config.saes,
        device=device
    ).to(device=device)
    vram_tracker.check("after moving SAETransformer to device")

    all_param_names = [name for name, _ in model.saes.named_parameters()]
    if config.saes.pretrained_sae_paths is not None:
        trainable_param_names = load_pretrained_saes(
            saes=model.saes,
            pretrained_sae_paths=config.saes.pretrained_sae_paths,
            all_param_names=all_param_names,
            retrain_saes=config.saes.retrain_saes,
        )
    else:
        trainable_param_names = all_param_names

    assert len(trainable_param_names) > 0, "No trainable parameters found."
    logger.info(f"Trainable parameters: {trainable_param_names}")

    vram_tracker.check("b4 call train()")
    train(
        config=config,
        model=model,
        train_loader=train_loader,
        trainable_param_names=trainable_param_names,
        device=device,
        run_names=run_names,
        wandb_wrapper=wandb_wrapper,
        cache_positions=cache_positions
    )
    if config.wandb_project:
        for queue in wandb_log_queues:
            queue.put("DONE")
        for wandb_process in wandb_procs:
            wandb_process.join()


if __name__ == "__main__":
    fire.Fire(main)
