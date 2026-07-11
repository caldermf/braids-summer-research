from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def confusion_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    probabilities = logits.softmax(-1)
    log_probabilities = logits.log_softmax(-1)
    true_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
    cross_entropy = -log_probabilities.gather(1, targets[:, None]).squeeze(1)
    entropy = -(probabilities * log_probabilities).sum(-1)
    top2 = probabilities.topk(2, dim=-1).values
    rank = 1 + (probabilities > true_probability[:, None]).sum(-1)
    return {
        "cross_entropy": cross_entropy,
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(probabilities.shape[-1]),
        "margin": top2[:, 0] - top2[:, 1],
        "true_probability": true_probability,
        "true_rank": rank,
        "correct": logits.argmax(-1).eq(targets),
    }


def brier_score(logits, targets):
    one_hot = F.one_hot(targets, logits.shape[-1]).float()
    return ((logits.softmax(-1) - one_hot) ** 2).sum(-1)

