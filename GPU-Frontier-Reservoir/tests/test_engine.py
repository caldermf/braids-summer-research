import torch

from gpu_frontier_reservoir.engine import projectivize_batch, scalar_identity_mask


def test_projectivize_shifts_each_matrix_independently():
    x = torch.zeros(2, 2, 2, 8, dtype=torch.int32)
    x[0, 0, 0, 3] = 1; x[0, 1, 1, 5] = 2
    x[1, 0, 1, 1] = 1
    y, pl, ends = projectivize_batch(x, 8)
    assert pl.tolist() == [3, 1]
    assert ends.tolist() == [6, 2]
    assert y[0, 0, 0, 0] == 1 and y[0, 1, 1, 2] == 2
    assert y[1, 0, 1, 0] == 1


def test_scalar_identity_requires_equal_diagonal_and_nonzero():
    x = torch.zeros(3, 2, 2, 4, dtype=torch.int32)
    x[0, 0, 0, 0] = x[0, 1, 1, 0] = 2
    x[1, 0, 0, 0] = 1; x[1, 1, 1, 0] = 2
    assert scalar_identity_mask(x).tolist() == [True, False, False]
