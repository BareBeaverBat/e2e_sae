import os
import time
from pathlib import Path
from queue import Empty
import multiprocessing as mp
from tempfile import TemporaryDirectory
from typing import Any, Literal, Optional

import wandb
import yaml

from e2e_sae.log import logger
from e2e_sae.scripts.train_tlens_saes.tlens_sae_train_config import Config, get_run_name
from e2e_sae.utils import init_wandb

wandb_save_flag = 'IS_CHECKPOINT'


# Most of the following method, as well as most of the concepts behind the rest of this file, are copied from
# https://github.com/bartbussmann/matryoshka_sae/blob/main/training.py
def new_wandb_process(config, log_queue, project, orig_base_run_name: str, run_name_suffix: str):
    run, config = init_wandb(config, project)
    updated_base_run_name = get_run_name(config)
    if orig_base_run_name != updated_base_run_name:
        logger.warning(f"for run name suffix {run_name_suffix}; mismatch between run name from config before "
                       f"init_wandb() and run name from config after: \n"
                       f"{orig_base_run_name}\nvs\n{updated_base_run_name}")
    run_name = updated_base_run_name + run_name_suffix
    run.name = run_name
    # Save the config to wandb
    with TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / f"final_config_{run_name_suffix}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(mode="json"), f, indent=2)
        wandb.save(str(config_path), policy="now", base_path=tmp_dir)
        # Unfortunately wandb.save is async, so we need to wait for it to finish before
        # continuing, and wandb python api provides no way to do this.
        # TODO: Find a better way to do this.
        time.sleep(1)

    while True:
        try:
            # Wait up to 1 second for new data
            log = log_queue.get(timeout=1)

            # Check for termination signal
            if log == "DONE":
                break

            assert isinstance(log, dict), f"wandb process for run {run_name} received malformed queue entry {log!r}"
            # Check if this is a checkpoint signal
            if log.get(wandb_save_flag):
                save_params = dict(log)  # note shallow copy
                save_params.pop(wandb_save_flag)
                # Create and log artifact
                wandb.save(**save_params)
            else:
                # Log regular metrics
                wandb.log(log['data'], log.get('step'))

        except Empty:
            continue

    wandb.finish()


class WandbWrapper:
    def __init__(self, wandb_log_queues: list[mp.Queue]):
        self.log_queues = wandb_log_queues

    def log(self, sae_variant_idx: int, data: dict[str, Any], step: int | None = None):
        assert 0 <= sae_variant_idx < len(self.log_queues)
        payload = {'data': data}
        if step is not None:
            payload['step'] = step
        self.log_queues[sae_variant_idx].put(payload)

    def save(self, sae_variant_idx: int, glob_str: str | os.PathLike | None = None,
             base_path: str | os.PathLike | None = None, policy: Optional[Literal["now", "live", "end"]] = None):
        assert 0 <= sae_variant_idx < len(self.log_queues)
        payload = {wandb_save_flag: True}
        if glob_str is not None:
            payload['glob_str'] = glob_str
        if base_path is not None:
            payload['base_path'] = base_path
        if policy is not None:
            payload['policy'] = policy

        self.log_queues[sae_variant_idx].put(payload)


def create_wandb_procs_queues_wrappers(config: Config, orig_base_run_name: str, run_name_suffixes: list[str]
                                       ) -> tuple[list[mp.Process], list[mp.Queue], WandbWrapper]:
    project = config.wandb_project
    log_queues = [mp.Queue() for _ in range(len(config.saes.sae_specs))]
    wandb_procs = [mp.Process(
        target=new_wandb_process, args=(config, queue, project, orig_base_run_name, run_name_suffixes[variant_idx])
    ) for variant_idx, queue in enumerate(log_queues)]
    for w_proc in wandb_procs:
        w_proc.start()

    wrapper = WandbWrapper(log_queues)

    return wandb_procs, log_queues, wrapper
