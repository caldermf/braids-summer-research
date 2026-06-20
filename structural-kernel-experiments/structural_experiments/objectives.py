from __future__ import annotations

from crispr_transformer.exact import make_evaluator

from .commutator_exact import make_commutator_evaluator


def make_objective_evaluator(
    *,
    metadata: dict,
    backend: str,
    device: str,
    batch_size: int,
    max_length: int,
):
    objective = metadata.get("objective", "ordinary_projlen")
    common = {
        "p": int(metadata["p"]),
        "n": int(metadata["n"]),
        "backend": backend,
        "device": device,
        "batch_size": batch_size,
    }
    if objective == "ordinary_projlen":
        return make_evaluator(**common)
    if objective == "commutator_projlen":
        generator_index = metadata.get("generator_index")
        if generator_index is None:
            raise ValueError("commutator objective requires generator_index")
        return make_commutator_evaluator(
            **common,
            generator_index=int(generator_index),
            max_length=max_length,
        )
    raise ValueError(f"unknown checkpoint objective: {objective!r}")
