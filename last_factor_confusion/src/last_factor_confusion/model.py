from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    p: int
    matrix_size: int = 3
    num_classes: int = 22
    d_model: int = 256
    heads: int = 8
    local_layers: int = 2
    global_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.08
    max_relative_degree: int = 512
    auxiliary_descents: bool = True

    def as_dict(self):
        return asdict(self)


class LastFactorTransformer(nn.Module):
    """Hierarchical coefficient-matrix then polynomial-degree transformer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d, m = config.d_model, config.matrix_size
        self.value_embedding = nn.Embedding(config.p, d)
        self.row_embedding = nn.Embedding(m, d)
        self.column_embedding = nn.Embedding(m, d)
        self.local_summary = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.matrix_summary = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.relative_degree = nn.Embedding(config.max_relative_degree + 1, d)
        self.gap_embedding = nn.Embedding(config.max_relative_degree + 1, d)
        layer = lambda: nn.TransformerEncoderLayer(
            d_model=d, nhead=config.heads, dim_feedforward=d * config.ffn_mult,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(layer(), config.local_layers, nn.LayerNorm(d))
        self.global_encoder = nn.TransformerEncoder(layer(), config.global_layers, nn.LayerNorm(d))
        self.output_norm = nn.LayerNorm(d)
        self.factor_head = nn.Linear(d, config.num_classes)
        self.descent_head = nn.Linear(d, 6) if config.auxiliary_descents else None
        rows = torch.arange(m).repeat_interleave(m)
        cols = torch.arange(m).repeat(m)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)

    def forward(self, coefficients, degree_mask, degrees=None):
        """
        coefficients: [B,D,M,M], residues in [0,p)
        degree_mask: [B,D], True for tokens retained by dense/sparse collation
        degrees: [B,D], exact relative degrees (defaults to 0..D-1)
        """
        if coefficients.ndim != 4:
            raise ValueError("coefficients must have shape [B,D,M,M]")
        b, depth, m, m2 = coefficients.shape
        if (m, m2) != (self.config.matrix_size, self.config.matrix_size):
            raise ValueError("matrix size does not match model configuration")
        if coefficients.min() < 0 or coefficients.max() >= self.config.p:
            raise ValueError("coefficient outside configured finite field")
        flat = coefficients.reshape(b * depth, m * m)
        tokens = self.value_embedding(flat)
        tokens = tokens + self.row_embedding(self.rows)[None] + self.column_embedding(self.cols)[None]
        summary = self.local_summary.expand(b * depth, -1, -1)
        local = self.local_encoder(torch.cat((summary, tokens), dim=1))[:, 0]
        local = local.reshape(b, depth, -1)
        if degrees is None:
            degrees = torch.arange(depth, device=coefficients.device)[None].expand(b, -1)
        degrees = degrees.clamp(0, self.config.max_relative_degree)
        gaps = torch.zeros_like(degrees)
        gaps[:, 1:] = (degrees[:, 1:] - degrees[:, :-1]).clamp_min(0)
        gaps = gaps.clamp_max(self.config.max_relative_degree)
        degree_tokens = local + self.relative_degree(degrees) + self.gap_embedding(gaps)
        matrix_token = self.matrix_summary.expand(b, -1, -1)
        global_tokens = torch.cat((matrix_token, degree_tokens), dim=1)
        padding = torch.cat((torch.zeros(b, 1, dtype=torch.bool, device=degree_mask.device), ~degree_mask), 1)
        encoded = self.global_encoder(global_tokens, src_key_padding_mask=padding)[:, 0]
        encoded = self.output_norm(encoded)
        return self.factor_head(encoded), None if self.descent_head is None else self.descent_head(encoded)

