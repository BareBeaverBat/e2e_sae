import math
from pathlib import Path
from typing import Optional, Self, Annotated, Literal, Any, TypeVar

import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator, BeforeValidator, \
    NonNegativeInt, NonNegativeFloat, conint, field_validator

from e2e_sae.data import DatasetConfig
from e2e_sae.log import logger
from e2e_sae.losses import LossConfigs
from e2e_sae.models.sae_impls import SAEInstantiationConfig
from e2e_sae.models.sparsifiers import SAE_Type
from e2e_sae.types import RootPath


class SAESpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: SAE_Type = Field(
        "Vanilla", description="The type of SAE to train. E.g. 'Vanilla', 'BatchTopK'"
    )
    is_matryoshka: bool = Field(
        False, description="Whether the SAE should use the (Bussman'24-style) Matryoshka training method."
    )
    dict_size_modifier: PositiveFloat = Field(
        1.0,
        description="Multiplicative modifier on the size of the dictionary (relative to the baseline ratio of dict size"
                    " to SAE input size that's specified for all SAE's in the run)"
    )
    top_k: Optional[PositiveInt]
    matryoshka_group_proportions: list[PositiveFloat] | None = Field(
        description="If is_matryoshka, list of fractions that add to 1 (describing the jumps between the sizes of the "
                    "nested groups of latents, as fractions of total number of latents in dictionary)"
    )
    k_aux: PositiveInt = Field(512, description="max number of dead latents to use in auxiliary loss")
    aux_coeff: Annotated[float, conint(ge=0)] = Field(
        0, description="coefficient for ghost-grads-like auxiliary loss term in loss function (e.g. 1/32), "
                       "most frequently used for TopK variants"
    )

    @model_validator(mode='after')
    def check_top_k(self) -> Self:
        top_k_needed = self.type in ['TopK', 'BatchTopK']
        if self.top_k is not None and not top_k_needed:
            logger.warning(f"top_k parameter should not be set for irrelevant SAE type {self.type}, value={self.top_k}")
        if self.top_k is None and top_k_needed:
            raise ValueError(f"top_k parameter should be set for SAE type {self.type}")
        return self

    @model_validator(mode='after')
    def check_matryoshka_group_proportions(self) -> Self:
        if not self.is_matryoshka and self.matryoshka_group_proportions is not None:
            logger.warning(f"matryoshka_group_proportions shouldn't be defined for an SAE that isn't using the "
                           f"Matryoshka training method; proportions: {self.matryoshka_group_proportions}")
        if self.is_matryoshka:
            if self.matryoshka_group_proportions is None or len(self.matryoshka_group_proportions) == 0:
                raise ValueError(f"matryoshka_group_proportions should be defined and non-empty for an SAE that is "
                                 f"using the Matryoshka training method, value={self.matryoshka_group_proportions}")
            elif abs(1.0-sum(self.matryoshka_group_proportions)) > 1e-6:
                raise ValueError(f"matryoshka_group_proportions should sum to 1: {self.matryoshka_group_proportions}")
        return self

SAE_LIST_T = TypeVar('SAE_LIST_T')

class SAEsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dict_size_to_input_ratio: PositiveFloat = 1.0
    pretrained_sae_paths: Annotated[
        list[RootPath] | None, BeforeValidator(lambda x: [x] if isinstance(x, str | Path) else x)
    ] = Field(None, description="Path to a pretrained SAEs to load. If None, don't load any.")
    retrain_saes: bool = Field(False, description="Whether to retrain the pretrained SAEs.")
    n_batches_to_dead: PositiveInt = Field(20, description="how many consecutive batches a latent can fail to ever "
                                                           "fire for before it's considered dead")
    sae_positions: Annotated[
        list[str], BeforeValidator(lambda x: [x] if isinstance(x, str) else x)
    ] = Field(
        ...,
        description="The names of the hook positions to train SAEs on. E.g. 'hook_resid_post' or "
        "['hook_resid_post', 'hook_mlp_out']. Each entry gets matched to all hook positions that "
        "contain the given string.",
    )
    sae_specs: list[SAESpecConfig] = Field(
        description="Specifications of the SAE variants to train.",
        default_factory=lambda: [SAESpecConfig()]
    )

    @classmethod
    @field_validator("sae_positions", "sae_specs", mode="after")
    def validate_sae_lists(cls, value: list[SAE_LIST_T]) -> list[SAE_LIST_T]:
        if len(value) == 0:
            raise ValueError(f"SAEsConfig must have non-empty lists of sae positions and specifications (specs)")
        return value


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None  # If None, don't log to Weights & Biases
    wandb_run_name: str | None = Field(
        None,
        description="If None, a run_name is generated based on (typically) important config "
        "parameters.",
    )
    wandb_run_name_prefix: str = Field("", description="Name that is prepended to the run name")
    seed: NonNegativeInt = Field(
        0,
        description="Seed set at start of script. Also used for train_data.seed and eval_data.seed "
        "if they are not set explicitly.",
    )
    tlens_model_name: str | None = None
    tlens_model_path: RootPath | None = Field(
        None,
        description="Path to '.pt' checkpoint. The directory housing this file should also contain "
        "'final_config.yaml' which is output by e2e_sae/scripts/train_tlens/run_train_tlens.py.",
    )
    tlens_model_dtype: str | None = Field(
        None, description="datatype to load the TransformerLens model in (e.g. float32, float16, or bfloat16),"
                          "overriding whatever the default would've been for the chosen model and loading method"
    )
    save_dir: RootPath | None = Path(__file__).parent / "out"
    n_samples: PositiveInt | None = None
    save_every_n_samples: PositiveInt | None
    eval_every_n_samples: PositiveInt | None = Field(
        None, description="If None, don't evaluate. If 0, only evaluate at the end."
    )
    eval_n_samples: PositiveInt | None
    batch_size: PositiveInt
    effective_batch_size: PositiveInt | None = None
    lr: PositiveFloat
    lr_schedule: Literal["linear", "cosine"] = "cosine"
    min_lr_factor: NonNegativeFloat = Field(
        0.1,
        description="The minimum learning rate as a factor of the initial learning rate. Used "
        "in the cooldown phase of a linear or cosine schedule.",
    )
    warmup_samples: NonNegativeInt = 0
    cooldown_samples: NonNegativeInt = 0
    max_grad_norm: PositiveFloat | None = None
    log_every_n_grad_steps: PositiveInt = 20
    collect_act_frequency_every_n_samples: NonNegativeInt = Field(
        20_000,
        description="Metrics such as activation frequency and alive neurons are calculated over "
        "fixed number of batches. This parameter specifies how often to calculate these metrics.",
    )
    act_frequency_n_tokens: PositiveInt = Field(
        100_000, description="The number of tokens to caclulate activation frequency metrics over."
    )
    loss: LossConfigs
    train_data: DatasetConfig
    eval_data: DatasetConfig | None = None
    saes: SAEsConfig

    @model_validator(mode="before")
    @classmethod
    def remove_deprecated_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Remove fields that are no longer used."""
        values.pop("collect_output_metrics_every_n_samples", None)
        return values

    @model_validator(mode="before")
    @classmethod
    def check_only_one_model_definition(cls, values: dict[str, Any]) -> dict[str, Any]:
        assert (values.get("tlens_model_name") is not None) + (
            values.get("tlens_model_path") is not None
        ) == 1, "Must specify exactly one of tlens_model_name or tlens_model_path."
        return values

    @model_validator(mode="after")
    def check_effective_batch_size(self) -> Self:
        if self.effective_batch_size is not None:
            assert (
                self.effective_batch_size % self.batch_size == 0
            ), "effective_batch_size must be a multiple of batch_size."
        return self

    @model_validator(mode="after")
    def verify_valid_eval_settings(self) -> Self:
        """User can't provide eval_every_n_samples without both eval_n_samples and data.eval."""
        if self.eval_every_n_samples is not None:
            assert (
                self.eval_n_samples is not None and self.eval_data is not None
            ), "Must provide eval_n_samples and data.eval when using eval_every_n_samples."
        return self

    @model_validator(mode="after")
    def cosine_schedule_requirements(self) -> Self:
        """Cosine schedule must have n_samples set in order to define the cosine curve."""
        if self.lr_schedule == "cosine":
            assert self.n_samples is not None, "Cosine schedule requires n_samples."
            assert self.cooldown_samples == 0, "Cosine schedule must not have cooldown_samples."
        return self


def determine_SAE_instantiation_conf(general_sae_configs: SAEsConfig, curr_sae_spec: SAESpecConfig,
                                     curr_sae_model_act_size: int) -> SAEInstantiationConfig:
    curr_dict_size = int(curr_sae_model_act_size*general_sae_configs.dict_size_to_input_ratio
                         * curr_sae_spec.dict_size_modifier)

    instantiation_conf = SAEInstantiationConfig(
        curr_sae_model_act_size, curr_dict_size, curr_sae_spec.type, curr_sae_spec.is_matryoshka,
        n_batches_to_dead=general_sae_configs.n_batches_to_dead, ghost_grads_aux_k=curr_sae_spec.k_aux,
        ghost_grads_aux_coeff=curr_sae_spec.aux_coeff, top_k=curr_sae_spec.top_k)

    if curr_sae_spec.is_matryoshka:
        instantiation_conf.matryoshka_group_sizes = distribute_to_integers(curr_sae_spec.matryoshka_group_proportions,
                                                                           curr_dict_size)

    return instantiation_conf


def distribute_to_integers(fractions: list[float], total: int) -> list[int]:
    products = [f * total for f in fractions]
    floored: list[int] = [math.floor(x) for x in products]
    current_sum = sum(floored)
    gap = total - current_sum
    if gap == 0:
        return floored

    remainders = [(x - math.floor(x), idx) for idx, x in enumerate(products)]
    # Sort by largest remainder
    remainders.sort(key=lambda r: r[0], reverse=True)
    # Distribute +1 to the top 'gap' remainders
    for i in range(gap):
        _, idx = remainders[i]
        floored[idx] += 1
    assert total == sum(floored)
    return floored


