from typing import Any, NamedTuple, Optional

import torch
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint

from e2e_sae.models.sae_impls import SAEForwardTertiaryResults
from e2e_sae.models.sparsifiers import SAE


class CacheActs(NamedTuple):
    input: Float[torch.Tensor, "... dim"]


class SAEActs(NamedTuple):
    input: Float[torch.Tensor, "... dim"]
    c: Float[torch.Tensor, "... c"]
    output: Float[torch.Tensor, "... dim"]
    tertiary_SAE_results: Optional[SAEForwardTertiaryResults] = None


def sae_hook(
    x: Float[torch.Tensor, "... dim"],
    hook: HookPoint | None,
    sae: SAE | torch.nn.Module,
    hook_acts: dict[str, Any],
    hook_key: str,
) -> Float[torch.Tensor, "... dim"]:
    """Runs the SAE on the input and stores the input, output and c in hook_acts under hook_key.

    Args:
        x: The input.
        hook: HookPoint object. Unused.
        sae: The SAE to run the input through.
        hook_acts: Dictionary of SAEActs and CacheActs objects to store the input, c, and output in.
        hook_key: The key in hook_acts to store the input, c, and output in.

    Returns:
        The output of the SAE.
    """
    output, c, tertiary_results = sae(x)
    hook_acts[hook_key] = SAEActs(input=x, c=c, output=output, tertiary_SAE_results=tertiary_results)
    return output


def cache_hook(
    x: Float[torch.Tensor, "... dim"],
    hook: HookPoint | None,
    hook_acts: dict[str, Any],
    hook_key: str,
) -> Float[torch.Tensor, "... dim"]:
    """Stores the input in hook_acts under hook_key.

    Args:
        x: The input.
        hook: HookPoint object. Unused.
        hook_acts: CacheActs object to store the input in.

    Returns:
        The input.
    """
    hook_acts[hook_key] = CacheActs(input=x)
    return x

def inject_hook(
        x: Float[torch.Tensor, "... dim"],
        hook: HookPoint | None,
        replacement_x: Float[torch.Tensor, "... dim"]
) -> Float[torch.Tensor, "... dim"]:
    """Replaces the actual data flowing through the model at the given hook point with the provided alternative tensor

    Args:
        x: the model-generated internal activations at the current hook position
        hook: a handle describing where in the model this hook method has been applied
        replacement_x: a different tensor that should overwrite the existing internal model activations at the current
            position

    Returns:
        The injected/replacement activations data
    """
    return replacement_x

