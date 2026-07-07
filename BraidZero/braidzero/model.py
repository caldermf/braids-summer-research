from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


PAD_TOKEN = 24
BOS_TOKEN = 25
VOCAB_SIZE = 26
ACTION_SIZE = 24


@dataclass
class BraidZeroModelConfig:
    max_len: int = 256
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    feature_dim: int = 6


class BraidZeroTransformer(nn.Module):
    """
    Transformer policy/value model for legal GNF prefixes.

    The model is deliberately not the verifier. It predicts which next simple
    factors are worth exact expansion, plus value heads for finite-shadow yield.
    """

    def __init__(self, config: BraidZeroModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len + 1, config.d_model)
        self.feature_mlp = nn.Sequential(
            nn.Linear(config.feature_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_mult * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.action_head = nn.Linear(config.d_model, ACTION_SIZE)
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 4),
        )

    def forward(self, tokens: torch.Tensor, features: torch.Tensor):
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [B,L], got {tuple(tokens.shape)}")
        if features.ndim != 2:
            raise ValueError(f"features must have shape [B,F], got {tuple(features.shape)}")
        batch, seq_len = tokens.shape
        if seq_len > self.config.max_len + 1:
            raise ValueError(f"sequence length {seq_len} exceeds model max {self.config.max_len + 1}")

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        hidden = self.token_emb(tokens) + self.pos_emb(positions)
        hidden[:, 0, :] = hidden[:, 0, :] + self.feature_mlp(features)
        pad_mask = tokens.eq(PAD_TOKEN)
        pad_mask[:, 0] = False
        hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
        pooled = self.norm(hidden[:, 0, :])
        return self.action_head(pooled), self.value_head(pooled)


def factors_to_tokens(factors: Sequence[int], *, max_len: int) -> list[int]:
    clipped = [int(x) for x in factors][-max_len:]
    tokens = [BOS_TOKEN] + clipped
    tokens.extend([PAD_TOKEN] * (max_len + 1 - len(tokens)))
    return tokens


def default_features(
    *,
    length: int,
    projlen: int,
    identity_defect: int,
    scalar_suffix_hits: int = 0,
    collision_hits: int = 0,
    bank_length: int = 0,
) -> list[float]:
    return [
        float(length) / 512.0,
        float(projlen) / 256.0,
        float(identity_defect) / 512.0,
        float(torch.log1p(torch.tensor(float(scalar_suffix_hits))).item()) / 10.0,
        float(torch.log1p(torch.tensor(float(collision_hits))).item()) / 10.0,
        float(bank_length) / 256.0,
    ]


def save_checkpoint(path: Path, model: BraidZeroTransformer, *, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> BraidZeroTransformer:
    payload = torch.load(path, map_location=map_location)
    config = BraidZeroModelConfig(**payload["config"])
    model = BraidZeroTransformer(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def score_legal_actions(
    model: BraidZeroTransformer,
    *,
    factors: Sequence[int],
    features: Sequence[float],
    legal_actions: Sequence[int],
    device: str | torch.device,
) -> list[tuple[int, float]]:
    model.eval()
    tokens = torch.tensor(
        [factors_to_tokens(factors, max_len=model.config.max_len)],
        dtype=torch.long,
        device=device,
    )
    feat = torch.tensor([list(features)], dtype=torch.float32, device=device)
    logits, values = model(tokens, feat)
    action_scores = logits[0].detach().cpu()
    value_bonus = float(values[0, 0].detach().cpu())
    scored = [(int(action), float(action_scores[int(action)]) + 0.05 * value_bonus) for action in legal_actions]
    return sorted(scored, key=lambda item: item[1], reverse=True)
