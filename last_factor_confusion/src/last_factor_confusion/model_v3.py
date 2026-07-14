from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelV3Config:
    p: int
    matrix_size: int = 3
    num_classes: int = 22
    d_model: int = 384
    heads: int = 6
    local_layers: int = 2
    global_layers: int = 8
    ffn_hidden: int = 1024
    dropout: float = 0.06
    auxiliary_descents: bool = True
    rope_base: float = 10000.0
    local_chunk_size: int = 32768

    def __post_init__(self):
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if (self.d_model // self.heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")

    def as_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        normalized = x * scale.to(x.dtype)
        return (normalized * self.weight).to(x.dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Rotate [B,H,L,D] queries/keys using exact, unbounded integer positions."""
    half = x.shape[-1] // 2
    frequency = torch.arange(half, device=x.device, dtype=torch.float32)
    frequency = base ** (-frequency / half)
    angle = positions.float()[:, None, :, None] * frequency[None, None, None, :]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    left, right = x[..., :half], x[..., half:]
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


class ExactDegreeAttention(nn.Module):
    def __init__(self, config: ModelV3Config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.d_model // config.heads
        self.rope_base = config.rope_base
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, positions, valid_mask):
        b, length, width = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        q = apply_rope(self.q_norm(q), positions, self.rope_base)
        k = apply_rope(self.k_norm(k), positions, self.rope_base)
        additive_mask = torch.zeros(b, 1, 1, length, device=x.device, dtype=q.dtype)
        additive_mask.masked_fill_(~valid_mask[:, None, None, :], torch.finfo(q.dtype).min)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=additive_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(b, length, width)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.gate_value = nn.Linear(width, 2 * hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, value = self.gate_value(x).chunk(2, dim=-1)
        return self.output(self.dropout(F.silu(gate) * value))


class GlobalBlock(nn.Module):
    def __init__(self, config: ModelV3Config):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = ExactDegreeAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, positions, valid_mask):
        x = x + self.dropout(self.attention(self.attention_norm(x), positions, valid_mask))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class ContinuousDegreeFeatures(nn.Module):
    """Non-table features that remain defined at degrees never seen during training."""
    def __init__(self, width: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(12, width), nn.SiLU(), nn.Linear(width, width, bias=False)
        )

    def forward(self, degrees, mask):
        d = degrees.float()
        previous = torch.cat((d[:, :1], d[:, :-1]), dim=1)
        following = torch.cat((d[:, 1:], d[:, -1:]), dim=1)
        left_gap = (d - previous).clamp_min(0)
        right_gap = (following - d).clamp_min(0)
        first = d[:, :1]
        lengths = mask.long().sum(1).clamp_min(1)
        last_index = lengths - 1
        last = d.gather(1, last_index[:, None])
        span = (last - first).clamp_min(1)
        relative = (d - first) / span
        features = torch.stack((
            torch.log1p(d), d / (1.0 + d),
            torch.log1p(left_gap), left_gap / (1.0 + left_gap),
            torch.log1p(right_gap), right_gap / (1.0 + right_gap),
            left_gap.eq(1).float(), right_gap.eq(1).float(),
            relative, 1.0 - relative,
            torch.arange(d.shape[1], device=d.device)[None].eq(0).expand_as(d).float(),
            torch.arange(d.shape[1], device=d.device)[None].eq(last_index[:, None]).float(),
        ), dim=-1)
        return self.project(features) * mask.unsqueeze(-1)


class LastFactorTransformerV3(nn.Module):
    """Sparse polynomial-matrix transformer with exact-degree attention."""
    architecture = "exact_degree_v3"

    def __init__(self, config: ModelV3Config):
        super().__init__()
        self.config = config
        d, m = config.d_model, config.matrix_size
        self.value_embedding = nn.Embedding(config.p, d)
        self.row_embedding = nn.Embedding(m, d)
        self.column_embedding = nn.Embedding(m, d)
        self.local_summary = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        local_layer = lambda: nn.TransformerEncoderLayer(
            d, config.heads, config.ffn_hidden, config.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(
            local_layer(), config.local_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False
        )
        self.degree_features = ContinuousDegreeFeatures(d)
        self.matrix_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.global_blocks = nn.ModuleList(GlobalBlock(config) for _ in range(config.global_layers))
        self.final_norm = RMSNorm(d)
        self.pool_query = nn.Parameter(torch.randn(d) * 0.02)
        self.pool_fusion = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.factor_head = nn.Linear(d, config.num_classes)
        self.descent_head = nn.Linear(d, 6) if config.auxiliary_descents else None
        self.register_buffer("rows", torch.arange(m).repeat_interleave(m), persistent=False)
        self.register_buffer("cols", torch.arange(m).repeat(m), persistent=False)

    def forward(self, coefficients, degree_mask, degrees=None):
        if coefficients.ndim != 4:
            raise ValueError("coefficients must have shape [B,D,M,M]")
        b, depth, m, m2 = coefficients.shape
        if (m, m2) != (self.config.matrix_size, self.config.matrix_size):
            raise ValueError("matrix size does not match model configuration")
        if coefficients.numel() and (coefficients.min() < 0 or coefficients.max() >= self.config.p):
            raise ValueError("coefficient outside configured finite field")
        if degrees is None:
            degrees = torch.arange(depth, device=coefficients.device)[None].expand(b, -1)
        flat = coefficients.reshape(b * depth, m * m)
        entries = self.value_embedding(flat)
        entries = entries + self.row_embedding(self.rows)[None] + self.column_embedding(self.cols)[None]
        summary = self.local_summary.expand(b * depth, -1, -1)
        local_input = torch.cat((summary, entries), dim=1)
        chunk_size = self.config.local_chunk_size
        local = torch.cat([
            self.local_encoder(local_input[start:start + chunk_size])[:, 0]
            for start in range(0, len(local_input), chunk_size)
        ]).reshape(b, depth, -1)
        degree_tokens = local + self.degree_features(degrees, degree_mask)
        special = self.matrix_token.expand(b, -1, -1)
        x = torch.cat((special, degree_tokens), dim=1)
        positions = torch.cat((torch.zeros_like(degrees[:, :1]), degrees + 1), dim=1)
        valid = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=degree_mask.device), degree_mask), dim=1)
        for block in self.global_blocks:
            x = block(x, positions, valid)
        x = self.final_norm(x)
        degree_x = x[:, 1:]
        scores = (degree_x * self.pool_query).sum(-1) / self.config.d_model ** 0.5
        scores = scores.masked_fill(~degree_mask, torch.finfo(scores.dtype).min)
        attention_pool = (degree_x * scores.softmax(1).unsqueeze(-1)).sum(1)
        last_index = degree_mask.long().sum(1).clamp_min(1) - 1
        last = degree_x.gather(1, last_index[:, None, None].expand(-1, 1, degree_x.shape[-1])).squeeze(1)
        pooled = self.pool_fusion(torch.cat((x[:, 0], attention_pool, degree_x[:, 0], last), dim=-1))
        logits = self.factor_head(pooled)
        return logits, None if self.descent_head is None else self.descent_head(pooled)
