import torch

from last_factor_confusion.data import collate_prefixes
from last_factor_confusion.model_v3 import (
    LastFactorTransformerV3,
    ModelV3Config,
    apply_rope,
)


def records():
    base = {"target_descents": [0] * 6, "trajectory_id": "t", "status": "clean"}
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    zero = [[0] * 3 for _ in range(3)]
    return [
        {**base, "matrix": [identity], "target_class": 0},
        {**base, "matrix": [identity, zero, [[0, 1, 0], [0, 0, 0], [0, 0, 0]]], "target_class": 1},
    ]


def tiny_model():
    return LastFactorTransformerV3(ModelV3Config(
        p=5, d_model=48, heads=3, local_layers=1, global_layers=2,
        ffn_hidden=96, dropout=0.0,
    ))


def test_v3_dense_and_sparse_shapes_and_finite_values():
    model = tiny_model().eval()
    for sparse in (False, True):
        x, mask, degrees, *_ = collate_prefixes(records(), sparse=sparse)
        logits, descents = model(x, mask, degrees)
        assert logits.shape == (2, 22)
        assert descents.shape == (2, 6)
        assert torch.isfinite(logits).all()


def test_rope_accepts_degrees_far_beyond_training_range():
    x = torch.randn(2, 3, 4, 16)
    positions = torch.tensor([[0, 1, 1000, 100000], [0, 7, 511, 999999]])
    result = apply_rope(x, positions, 10000.0)
    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert not torch.allclose(result[:, :, 2], result[:, :, 3])


def test_padding_does_not_change_unpadded_prediction():
    model = tiny_model().eval()
    x, mask, degrees, *_ = collate_prefixes(records()[:1], sparse=True)
    expected = model(x, mask, degrees)[0]
    padded_x = torch.cat((x, torch.zeros(1, 3, 3, 3, dtype=x.dtype)), dim=1)
    padded_mask = torch.cat((mask, torch.zeros(1, 3, dtype=torch.bool)), dim=1)
    padded_degrees = torch.cat((degrees, torch.tensor([[5000, 7000, 9000]])), dim=1)
    actual = model(padded_x, padded_mask, padded_degrees)[0]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_v3_backward_reaches_global_attention():
    model = tiny_model().train()
    x, mask, degrees, labels, *_ = collate_prefixes(records(), sparse=True)
    loss = torch.nn.functional.cross_entropy(model(x, mask, degrees)[0], labels)
    loss.backward()
    gradient = model.global_blocks[0].attention.qkv.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
