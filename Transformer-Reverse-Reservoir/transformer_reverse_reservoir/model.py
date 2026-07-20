from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from braidzero.core import sha256_file
from last_factor_confusion.data import collate_prefixes
from last_factor_confusion.factors import FactorTable
from last_factor_confusion.model_v3 import LastFactorTransformerV3, ModelV3Config


class LastFactorOracle:
    """Frozen v3 model with an explicit map between model classes and peyl factors."""

    def __init__(self, checkpoint: Path, calibration: Path, env, device: torch.device):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("architecture") != LastFactorTransformerV3.architecture:
            raise ValueError("checkpoint is not an exact-degree-v3 model")
        config = ModelV3Config(**saved["model_config"])
        if config.p != env.p or config.matrix_size != env.dim:
            raise ValueError("checkpoint representation does not match the search environment")

        self.model = LastFactorTransformerV3(config).to(device).eval()
        self.model.load_state_dict(saved["state_dict"])
        self.device = device
        self.env = env
        self.temperature = float(json.loads(calibration.read_text())["temperature"])

        table = FactorTable.from_peyl(env.n)
        factor_to_class: dict[int, int] = {}
        for factor_id, permutation in enumerate(env.nf_table.divs):
            try:
                factor_to_class[factor_id] = table.class_id(permutation)
            except ValueError:
                pass
        if len(factor_to_class) != config.num_classes:
            raise RuntimeError(
                f"expected {config.num_classes} proper factors, got {len(factor_to_class)}"
            )
        self.factor_to_class = factor_to_class
        self.class_to_factor = {
            class_id: factor_id for factor_id, class_id in factor_to_class.items()
        }
        self.proper_factor_ids = tuple(
            self.class_to_factor[class_id] for class_id in range(config.num_classes)
        )
        self.metadata = {
            "architecture": saved["architecture"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_checksum": sha256_file(checkpoint),
            "calibration": str(calibration.resolve()),
            "calibration_checksum": sha256_file(calibration),
            "temperature": self.temperature,
            "num_classes": config.num_classes,
        }

    def matrix_record(self, image: np.ndarray) -> dict:
        normalized = self.env.polymat.projectivise(np.asarray(image)) % self.env.p
        return {
            "matrix": np.moveaxis(normalized, -1, 0).tolist(),
            "target_class": 0,
            "target_descents": [0] * 6,
            "trajectory_id": "reverse-search",
            "status": "clean",
        }

    @torch.no_grad()
    def logits(self, images: Sequence[np.ndarray], batch_size: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            records = [self.matrix_record(image) for image in images[start : start + batch_size]]
            coefficients, mask, degrees, _, _, _ = collate_prefixes(records, sparse=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                output, _ = self.model(
                    coefficients.to(self.device),
                    mask.to(self.device),
                    degrees.to(self.device),
                )
            chunks.append(output.float().cpu())
        if not chunks:
            return torch.empty((0, len(self.class_to_factor)), dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    def legal_distribution(
        self, logits: torch.Tensor, legal_factor_ids: Sequence[int]
    ) -> tuple[dict[int, float], dict[int, int], float]:
        """Return conditional log probabilities, ranks, and entropy over legal factors."""
        legal_factor_ids = tuple(int(x) for x in legal_factor_ids)
        classes = torch.tensor(
            [self.factor_to_class[factor_id] for factor_id in legal_factor_ids],
            dtype=torch.long,
        )
        legal_logits = logits[classes] / self.temperature
        legal_log_probs = F.log_softmax(legal_logits, dim=0)
        probabilities = legal_log_probs.exp()
        order = torch.argsort(legal_log_probs, descending=True).tolist()
        ranks = {legal_factor_ids[index]: rank + 1 for rank, index in enumerate(order)}
        log_probs = {
            factor_id: float(legal_log_probs[index])
            for index, factor_id in enumerate(legal_factor_ids)
        }
        entropy = float(-(probabilities * legal_log_probs).sum())
        return log_probs, ranks, entropy
