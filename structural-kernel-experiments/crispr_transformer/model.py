from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    p: int
    n: int = 4
    factor_vocab_size: int = 24
    max_length: int = 96
    max_delete: int = 16
    max_insert: int = 16
    max_net_delta: int = 3
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    action_dim: int = 32
    dropout: float = 0.10

    def to_dict(self) -> dict:
        return asdict(self)


class GeometryTransformer(nn.Module):
    """Score variable-length edit geometries for complete legal GNF words."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.factor_embedding = nn.Embedding(
            config.factor_vocab_size + 1,
            config.d_model,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        self.history_projection = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.Tanh(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.delete_embedding = nn.Embedding(config.max_delete + 1, config.action_dim)
        self.insert_embedding = nn.Embedding(config.max_insert + 1, config.action_dim)
        self.delta_embedding = nn.Embedding(
            2 * config.max_net_delta + 1,
            config.action_dim,
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(3, config.action_dim),
            nn.GELU(),
        )
        input_width = 3 * config.d_model + 4 * config.action_dim
        self.scorer = nn.Sequential(
            nn.Linear(input_width, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def encode(self, tokens, histories, lengths):
        batch_size, width = tokens.shape
        if width > self.config.max_length:
            raise ValueError(
                f"sequence length {width} exceeds model maximum {self.config.max_length}"
            )
        positions = torch.arange(width, device=tokens.device)[None, :]
        mask = positions >= lengths[:, None]
        scale = torch.clamp(lengths.to(histories.dtype), min=1.0)[:, None]
        normalized_history = histories / (4.0 * scale)
        hidden = (
            self.factor_embedding(tokens)
            + self.position_embedding(positions)
            + self.history_projection(normalized_history[..., None])
        )
        hidden = self.final_norm(self.encoder(hidden, src_key_padding_mask=mask))
        keep = (~mask).to(hidden.dtype)[..., None]
        pooled = (hidden * keep).sum(dim=1) / lengths.clamp(min=1)[:, None]
        return hidden, pooled

    def score_encoded(self, hidden, pooled, lengths, action_parents, actions):
        starts = actions[:, 0]
        deletes = actions[:, 1]
        inserts = actions[:, 2]
        ends = starts + deletes - 1
        start_hidden = hidden[action_parents, starts]
        end_hidden = hidden[action_parents, ends]
        global_hidden = pooled[action_parents]
        parent_lengths = lengths[action_parents].to(hidden.dtype)
        geometry_features = torch.stack(
            (
                starts.to(hidden.dtype) / parent_lengths,
                deletes.to(hidden.dtype) / parent_lengths,
                inserts.to(hidden.dtype) / parent_lengths,
            ),
            dim=1,
        )
        delta_index = inserts - deletes + self.config.max_net_delta
        features = torch.cat(
            (
                start_hidden,
                end_hidden,
                global_hidden,
                self.delete_embedding(deletes),
                self.insert_embedding(inserts),
                self.delta_embedding(delta_index),
                self.geometry_projection(geometry_features),
            ),
            dim=1,
        )
        return self.scorer(features).squeeze(1)

    def forward(self, tokens, histories, lengths, action_parents, actions):
        hidden, pooled = self.encode(tokens, histories, lengths)
        return self.score_encoded(hidden, pooled, lengths, action_parents, actions)


def load_model(path: str, device: str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "crispr-transformer-geometry-model-v1":
        raise ValueError("not a CRISPR-Transformer geometry checkpoint")
    config = ModelConfig(**payload["model_config"])
    model = GeometryTransformer(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload
