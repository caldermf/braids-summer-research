import torch

from last_factor_confusion.data import collate_prefixes
from last_factor_confusion.metrics import confusion_metrics
from last_factor_confusion.model import LastFactorTransformer, ModelConfig


def records():
    base = {"target_descents": [0] * 6, "trajectory_id": "t", "status": "clean"}
    return [
        {**base, "matrix": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], "target_class": 0},
        {**base, "matrix": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 0], [0, 0, 0]]], "target_class": 1},
    ]


def test_dense_and_sparse_forward():
    model = LastFactorTransformer(ModelConfig(p=5, d_model=32, heads=4, local_layers=1, global_layers=1))
    for sparse in (False, True):
        x, mask, degree, target, _, _ = collate_prefixes(records(), sparse=sparse)
        logits, descents = model(x, mask, degree)
        assert logits.shape == (2, 22)
        assert descents.shape == (2, 6)
        assert confusion_metrics(logits, target)["true_rank"].shape == (2,)


def test_internal_zero_degree_is_preserved_dense():
    rec = {**records()[0], "matrix": [records()[0]["matrix"][0], [[0]*3 for _ in range(3)], records()[0]["matrix"][0]]}
    _, mask, degrees, *_ = collate_prefixes([rec], sparse=False)
    assert mask.tolist() == [[True, True, True]]
    assert degrees.tolist() == [[0, 1, 2]]

