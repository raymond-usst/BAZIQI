"""
DeepSeek-style Multi-Head Latent Attention (MLA) Transformer backbone.

Key innovations:
- MLA: KV heads compressed into low-rank latent for memory efficiency
- DeepNorm: alpha/beta scaling for stable training at extreme depth
- SwiGLU: Gated FFN (as in LLaMA / DeepSeek)
- RoPE: Rotary positional embeddings for spatial awareness
- Patch embedding: Convert 2D board → token sequence
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ai.log_utils import get_logger

_log = get_logger(__name__)


# ============================================================
#  Rotary Position Embedding
# ============================================================

class RotaryEmbedding(nn.Module):
    """Rotary positional embedding for 2D spatial positions."""

    def __init__(self, dim: int, max_seq_len: int = 256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


def rotate_half(x):
    """Rotate half the hidden dims of x."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embedding to query/key tensors."""
    return x * cos + rotate_half(x) * sin


# ============================================================
#  Multi-Head Latent Attention (MLA)
# ============================================================

class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek-V2 style MLA.

    Instead of projecting x → K, V per head independently:
      c_kv = W_dkv(x)            # compress to low-rank latent
      K = W_uk(c_kv)             # reconstruct keys
      V = W_uv(c_kv)             # reconstruct values
      Q = W_q(x)                 # queries as normal

    During inference, only `c_kv` needs to be cached (much smaller than full K,V).
    """

    def __init__(self, d_model: int, n_heads: int, d_kv_compress: int,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_kv_compress = d_kv_compress

        # Query projection (standard)
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        # KV compression: x → low-rank latent
        self.W_dkv = nn.Linear(d_model, d_kv_compress, bias=False)

        # KV reconstruction: latent → full K, V
        self.W_uk = nn.Linear(d_kv_compress, d_model, bias=False)
        self.W_uv = nn.Linear(d_kv_compress, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x: torch.Tensor,
                rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape

        # Queries
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Compress KV into latent
        c_kv = self.W_dkv(x)  # (B, L, d_kv_compress)

        # Reconstruct K, V from latent
        K = self.W_uk(c_kv).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_uv(c_kv).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K
        if rope_cos is not None and rope_sin is not None:
            cos = rope_cos.unsqueeze(1)  # (1, 1, L, d_head)
            sin = rope_sin.unsqueeze(1)
            Q = apply_rotary_emb(Q, cos, sin)
            K = apply_rotary_emb(K, cos, sin)

        # Attention — use Flash/Memory-Efficient kernel via SDPA (PyTorch 2.0+)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )  # (B, n_heads, L, d_head)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


# ============================================================
#  SwiGLU Feed-Forward Network
# ============================================================

class SwiGLUFFN(nn.Module):
    """Gated FFN with SiLU activation (SwiGLU), as in LLaMA / DeepSeek."""

    def __init__(self, d_model: int, ffn_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w_up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w_down = nn.Linear(ffn_hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


# ============================================================
#  DeepNorm Transformer Block
# ============================================================

class DeepNormTransformerBlock(nn.Module):
    """
    Transformer block with DeepNorm for ultra-deep stability.

    DeepNorm formula:
      x_out = LayerNorm(x * alpha + Sublayer(x))

    Initialization: residual weights scaled by beta.
    With proper alpha/beta, this is stable beyond 1000 layers.
    """

    def __init__(self, d_model: int, n_heads: int, d_kv_compress: int,
                 ffn_hidden: int, dropout: float = 0.0,
                 alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

        # Attention sublayer
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, n_heads, d_kv_compress, dropout)

        # FFN sublayer
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_hidden, dropout)

    def forward(self, x: torch.Tensor,
                rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        # DeepNorm attention: LN(alpha * x + Attn(x))
        normed = self.norm_attn(x)
        attn_out = self.attn(normed, rope_cos, rope_sin)
        x = self.alpha * x + attn_out

        # DeepNorm FFN: LN(alpha * x + FFN(x))
        normed = self.norm_ffn(x)
        ffn_out = self.ffn(normed)
        x = self.alpha * x + ffn_out

        return x


# ============================================================
#  Patch Embedding
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Convert board observation to token sequence.

    Input: (B, C, H, W) board
    Output: (B, num_patches + 1, d_model) — patches + CLS token
    """

    def __init__(self, in_channels: int = 4, d_model: int = 256,
                 patch_size: int = 3, view_size: int = 21):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (view_size // patch_size) ** 2  # 7*7=49 for 21/3
        self.remainder = view_size % patch_size

        # Use conv to extract patch features
        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Trim if not divisible by patch_size
        if self.remainder > 0:
            trim = self.remainder
            x = x[:, :, :x.shape[2] - trim, :x.shape[3] - trim]

        # (B, C, H, W) → (B, d_model, H/p, W/p) → (B, num_patches, d_model)
        patches = self.proj(x)
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)  # (B, N+1, d_model)

        return self.layer_norm(tokens)


# ============================================================
#  Full Transformer Backbone
# ============================================================

class TransformerBackbone(nn.Module):
    """
    DeepSeek MLA Transformer backbone for board game representation.

    Input: (B, obs_channels, view_size, view_size)
    Output: (B, hidden_state_dim) — CLS token as state embedding
    """

    def __init__(self, obs_channels: int = 4, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 12,
                 d_kv_compress: int = 64, ffn_hidden: int = 512,
                 hidden_state_dim: int = 128,
                 patch_size: int = 3, view_size: int = 21,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_state_dim = hidden_state_dim

        # Patch embedding
        num_patches = (view_size // patch_size) ** 2
        seq_len = num_patches + 1  # +1 for CLS
        self.patch_embed = PatchEmbedding(obs_channels, d_model, patch_size, view_size)

        # RoPE
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len=seq_len + 16)

        # DeepNorm alpha: (2 * n_layers)^0.25 per the paper
        # BUT: Our implementation is Pre-LN, so alpha > 1 causes exponential growth of x.
        # We revert to standard Pre-LN (alpha=1.0) for stability.
        # alpha = (2 * n_layers) ** 0.25
        alpha = 1.0

        # Transformer layers
        self.layers = nn.ModuleList([
            DeepNormTransformerBlock(
                d_model, n_heads, d_kv_compress, ffn_hidden, dropout, alpha
            )
            for _ in range(n_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(d_model)

        # Project CLS token to hidden state
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, hidden_state_dim),
        )

        # Initialize with DeepNorm beta scaling
        beta = (8 * n_layers) ** -0.25
        self._deepnorm_init(beta)

    def _deepnorm_init(self, beta: float):
        """Scale residual branch weights by beta for DeepNorm stability."""
        for layer in self.layers:
            # Scale attention output projection
            with torch.no_grad():
                layer.attn.W_o.weight.mul_(beta)
                # Scale FFN output projection
                layer.ffn.w_down.weight.mul_(beta)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, C, H, W) board observation
        Returns:
            hidden_state: (B, hidden_state_dim)
        """
        # Patch embed
        tokens = self.patch_embed(obs)  # (B, L, d_model)
        L = tokens.shape[1]

        # RoPE
        cos, sin = self.rope(L)

        # Transformer layers
        for layer in self.layers:
            tokens = layer(tokens, cos, sin)

        # Final norm + CLS token
        tokens = self.final_norm(tokens)
        cls_out = tokens[:, 0]  # CLS token

        # Project to hidden state
        hidden = self.head(cls_out)

        # Normalize to unit sphere (matches original design)
        hidden = hidden / (hidden.norm(dim=1, keepdim=True) + 1e-8)

        return hidden
