"""
Engram — External episodic memory module.

Stores (key, value) pairs from past game positions and retrieves relevant
memories via cross-attention, allowing the network to recall patterns
from previously seen board states.

'Engram' = the neural trace left by an experience.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from ai.log_utils import get_logger

_log = get_logger(__name__)


class MemoryBank:
    """
    Fixed-capacity memory bank storing hidden state embeddings and associated data.

    Stored entries:
        key: (hidden_state_dim,) — hidden state embedding
        value: (value_dim,) — concatenation of [policy_summary, value, reward]
        priority: float — importance score (higher for winning/critical positions)
    """

    def __init__(self, capacity: int = 10000, key_dim: int = 128,
                 value_dim: int = 128):
        self.capacity = capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.count = 0
        self.write_idx = 0

        # Pre-allocated tensors (CPU for storage)
        self.keys = torch.zeros(capacity, key_dim)
        self.values = torch.zeros(capacity, value_dim)
        self.priorities = torch.zeros(capacity)

    def write(self, keys: torch.Tensor, values: torch.Tensor,
              priorities: Optional[torch.Tensor] = None):
        """
        Write batch of entries to memory.

        Args:
            keys: (N, key_dim)
            values: (N, value_dim)
            priorities: (N,) importance scores
        """
        keys = keys.detach().cpu()
        values = values.detach().cpu()
        N = keys.shape[0]

        if priorities is None:
            priorities = torch.ones(N)
        else:
            priorities = priorities.detach().cpu()

        for i in range(N):
            idx = self.write_idx % self.capacity
            self.keys[idx] = keys[i]
            self.values[idx] = values[i]
            self.priorities[idx] = priorities[i]
            self.write_idx += 1
            self.count = min(self.count + 1, self.capacity)

    def read(self, queries: torch.Tensor, top_k: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k most similar entries for each query.

        Args:
            queries: (B, key_dim) query vectors
            top_k: number of entries to retrieve

        Returns:
            retrieved_keys: (B, top_k, key_dim)
            retrieved_values: (B, top_k, value_dim)
        """
        if self.count == 0:
            B = queries.shape[0]
            device = queries.device
            return (torch.zeros(B, top_k, self.key_dim, device=device),
                    torch.zeros(B, top_k, self.value_dim, device=device))

        device = queries.device
        valid_keys = self.keys[:self.count].to(device)     # (M, key_dim)
        valid_values = self.values[:self.count].to(device)  # (M, value_dim)
        valid_priorities = self.priorities[:self.count].to(device)  # (M,)

        # Cosine similarity
        queries_norm = F.normalize(queries, dim=-1)                  # (B, key_dim)
        keys_norm = F.normalize(valid_keys, dim=-1)                  # (M, key_dim)
        similarity = torch.matmul(queries_norm, keys_norm.t())       # (B, M)

        # Boost by priority
        similarity = similarity * (1.0 + 0.1 * valid_priorities.unsqueeze(0))

        # Top-k
        k = min(top_k, self.count)
        _, indices = similarity.topk(k, dim=-1)  # (B, k)

        retrieved_keys = valid_keys[indices]      # (B, k, key_dim)
        retrieved_values = valid_values[indices]   # (B, k, value_dim)

        # Pad if needed
        if k < top_k:
            pad_k = top_k - k
            B = queries.shape[0]
            retrieved_keys = torch.cat([
                retrieved_keys,
                torch.zeros(B, pad_k, self.key_dim, device=device)
            ], dim=1)
            retrieved_values = torch.cat([
                retrieved_values,
                torch.zeros(B, pad_k, self.value_dim, device=device)
            ], dim=1)

        return retrieved_keys, retrieved_values
    
    def size(self) -> int:
        return self.count

    def state_dict(self):
        return {
            'keys': self.keys[:self.count].clone(),
            'values': self.values[:self.count].clone(),
            'priorities': self.priorities[:self.count].clone(),
            'count': self.count,
            'write_idx': self.write_idx,
        }

    def load_state_dict(self, state):
        count = state['count']
        self.keys[:count] = state['keys']
        self.values[:count] = state['values']
        self.priorities[:count] = state['priorities']
        self.count = count
        self.write_idx = state['write_idx']

        # Sanitize NaN/Inf that may have been persisted from corrupted training
        if torch.isnan(self.keys[:count]).any() or torch.isinf(self.keys[:count]).any() or \
           torch.isnan(self.values[:count]).any() or torch.isinf(self.values[:count]).any():
            print("[MemoryBank] WARNING: Loaded state contains NaN/Inf! Resetting memory.")
            self.keys.zero_()
            self.values.zero_()
            self.priorities.zero_()
            self.count = 0
            self.write_idx = 0


class EngramModule(nn.Module):
    """
    Cross-attention retrieval from episodic memory.

    Given a hidden state query, retrieves top-k memories and attends
    over them to produce a memory-augmented representation.

    Final output = hidden_state + memory_output (residual connection)
    """

    def __init__(self, hidden_dim: int = 128, value_dim: int = 128,
                 n_heads: int = 4, top_k: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.value_dim = value_dim
        self.n_heads = n_heads
        self.top_k = top_k
        self.d_head = hidden_dim // n_heads

        # Query projection (from current hidden state)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Key projection (from retrieved keys)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Value projection (from retrieved values)
        self.W_v = nn.Linear(value_dim, hidden_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Gate: learn how much to mix memory vs current state
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scale = self.d_head ** -0.5

    def forward(self, hidden_state: torch.Tensor,
                memory_keys: torch.Tensor,
                memory_values: torch.Tensor) -> torch.Tensor:
        """
        Attend over retrieved memories and produce augmented hidden state.
        
        Args:
            hidden_state: (B, hidden_dim) — current state
            memory_keys: (B, top_k, hidden_dim) — retrieved memory keys
            memory_values: (B, top_k, value_dim) — retrieved memory values
            
        Returns:
            augmented_state: (B, hidden_dim) — memory-augmented state
        """
        B = hidden_state.shape[0]

        # Q from current state: (B, 1, n_heads, d_head)
        Q = self.W_q(hidden_state).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)

        # K from memory: (B, top_k, n_heads, d_head)
        K = self.W_k(memory_keys).view(B, self.top_k, self.n_heads, self.d_head).transpose(1, 2)

        # V from memory: (B, top_k, n_heads, d_head)
        V = self.W_v(memory_values).view(B, self.top_k, self.n_heads, self.d_head).transpose(1, 2)

        # Cross-attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, n_heads, 1, top_k)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # (B, n_heads, 1, d_head)
        out = out.transpose(1, 2).contiguous().view(B, self.hidden_dim)
        memory_out = self.W_o(out)

        # Gated residual: gate controls how much memory to incorporate
        gate_input = torch.cat([hidden_state, memory_out], dim=-1)
        g = self.gate(gate_input)
        augmented = hidden_state + g * memory_out

        return self.layer_norm(augmented)

    def create_value_embedding(self, policy_logits: torch.Tensor,
                               value: torch.Tensor,
                               reward: torch.Tensor) -> torch.Tensor:
        """
        Create a value embedding from prediction outputs for memory storage.
        
        Args:
            policy_logits: (B, action_size)
            value: (B,)
            reward: (B,)
            
        Returns:
            value_embedding: (B, value_dim)
        """
        # Compactly represent (value, reward) and perhaps max logit
        # (This simplistic concatenation is robust enough for retrieval context)
        B = value.shape[0]
        device = value.device
        
        # We assume value_dim >= 3. If smaller, error.
        out = torch.zeros(B, self.value_dim, device=device)
        out[:, 0] = value
        out[:, 1] = reward
        # Use max logit as a proxy for policy
        if policy_logits is not None:
             out[:, 2] = policy_logits.max(dim=1)[0]
        
        return out
