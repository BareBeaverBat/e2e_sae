from pathlib import Path

import torch
import yaml
from sae_lens.config import DTYPE_MAP
from transformer_lens import HookedTransformer, HookedTransformerConfig

from e2e_sae.scripts.train_tlens.run_train_tlens import HookedTransformerPreConfig
from e2e_sae.types import RootPath


def load_tlens_model(
    tlens_model_name: str | None, tlens_model_path: RootPath | None, tlens_model_dtype: str | None
) -> HookedTransformer:
    """Load transformerlens model from either HuggingFace or local path."""
    if tlens_model_name is not None:
        tlens_model = HookedTransformer.from_pretrained(tlens_model_name) if tlens_model_dtype is None \
            else HookedTransformer.from_pretrained(tlens_model_name, dtype=tlens_model_dtype)
    else:
        assert tlens_model_path is not None, "tlens_model_path is None."
        # Load the tlens_config
        assert (
            tlens_model_path.parent / "final_config.yaml"
        ).exists(), "final_config.yaml does not exist."
        with open(tlens_model_path.parent / "final_config.yaml") as f:
            tlens_config = HookedTransformerPreConfig(**yaml.safe_load(f)["tlens_config"])
        hooked_transformer_config = HookedTransformerConfig(**tlens_config.model_dump())
        if tlens_model_dtype:
            assert tlens_model_dtype in DTYPE_MAP, (f"tried to load local TransformerLens model with unsupported "
                                                    f"override dtype {tlens_model_dtype}")
            hooked_transformer_config.dtype = DTYPE_MAP[tlens_model_dtype]

        # Load the model
        tlens_model = HookedTransformer(hooked_transformer_config)
        tlens_model.load_state_dict(torch.load(tlens_model_path, map_location="cpu"))

    assert tlens_model.tokenizer is not None
    return tlens_model


def load_pretrained_saes(
    saes: torch.nn.ModuleDict,
    pretrained_sae_paths: list[Path],
    all_param_names: list[str],
    retrain_saes: bool,
) -> list[str]:
    """Load in the pretrained SAEs to saes (in place) and return the trainable param names.

    Args:
        saes: The base SAEs to load the pretrained SAEs into. Updated in place.
        pretrained_sae_paths: List of paths to the pretrained SAEs.
        all_param_names: List of all the parameter names in saes.
        retrain_saes: Whether to retrain the pretrained SAEs.

    Returns:
        The updated all_param_names.
    """
    pretrained_sae_params = {}
    for pretrained_sae_path in pretrained_sae_paths:
        # Add new sae params (note that this will overwrite existing SAEs with the same name)
        pretrained_sae_params = {**pretrained_sae_params, **torch.load(pretrained_sae_path)}
    sae_state_dict = {**dict(saes.named_parameters()), **pretrained_sae_params}

    saes.load_state_dict(sae_state_dict)
    if not retrain_saes:
        # Don't retrain the pretrained SAEs
        trainable_param_names = [
            name for name in all_param_names if name not in pretrained_sae_params
        ]
    else:
        trainable_param_names = all_param_names

    return trainable_param_names
