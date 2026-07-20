import random

import numpy as np

from transformer_reverse_reservoir.reservoir import LikelihoodReservoir, ReverseState


def state(index: int, nll: float) -> ReverseState:
    return ReverseState(
        residual=np.eye(1, dtype=np.int32)[..., None],
        suffix=(index + 1,), cumulative_nll=nll, edge_nll=nll,
        edge_rank=1, entropy=0.0, projlen=0, digest=str(index),
    )


def test_bucket_keys_are_unbounded():
    reservoir = LikelihoodReservoir(10, 0.25, random.Random(1))
    reservoir.add(state(0, 0.10))
    reservoir.add(state(1, 100.10))
    assert set(reservoir.buckets) == {0, 400}


def test_selection_contains_exploitation_and_exploration():
    reservoir = LikelihoodReservoir(100, 1.0, random.Random(2))
    for index in range(30):
        reservoir.add(state(index, float(index // 10)))
    selected, summary = reservoir.select(10, 0.6)
    assert len(selected) == 10
    assert summary["exploit"] == 6
    assert summary["explore"] == 4
    assert any(item.average_nll >= 1.0 for item in selected)

