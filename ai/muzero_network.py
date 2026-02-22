"""MuZero Neural Networks with DeepSeek MLA, Engram, and EfficientZero components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .transformer_backbone import TransformerBackbone
from .engram import EngramModule
from .consistency import ConsistencyModule
from .log_utils import get_logger

_log = get_logger(__name__)


class DynamicsNetwork(nn.Module):
    """
    g(hidden_state, action) → (next_hidden_state, reward)
    
    Predicts the next hidden state and immediate reward given current state + action.
    """
    def __init__(self, hidden_state_dim: int = 128, action_size: int = 441,
                 fc_hidden: int = 512):
        super().__init__()
        self.action_embed = nn.Linear(action_size, hidden_state_dim)
        
        # Larger dynamics model to match backbone capacity
        self.state_net = nn.Sequential(
            nn.Linear(hidden_state_dim * 2, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
            nn.Linear(fc_hidden, hidden_state_dim),
        )
        
        # Residual skip projection: ensures identity-like mapping by default
        self.skip_proj = nn.Linear(hidden_state_dim * 2, hidden_state_dim)
        
        self.reward_net = nn.Sequential(
            nn.Linear(hidden_state_dim * 2, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
            nn.Linear(fc_hidden, 1),
            # No Tanh for reward, let it learn range (or can clamp later)
        )
        self.hidden_state_dim = hidden_state_dim

    def forward(self, hidden_state: torch.Tensor,
                action_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_emb = self.action_embed(action_onehot)
        combined = torch.cat([hidden_state, action_emb], dim=1)
        
        next_state = self.state_net(combined) + self.skip_proj(combined)
        # Normalize hidden state to unit sphere
        next_state = next_state / (next_state.norm(dim=1, keepdim=True) + 1e-8)
        
        reward = self.reward_net(combined).squeeze(-1)
        return next_state, reward


class PredictionNetwork(nn.Module):
    """f(hidden_state) → (policy, value)
    
    Policy head uses row+col factored decomposition for 21×21 action space.
    Value and policy heads have separate representation transforms (Level 3)
    to decouple gradient flows and give each head a different learned view.
    """
    def __init__(self, hidden_state_dim: int = 128, action_size: int = 441,
                 fc_hidden: int = 512):
        super().__init__()
        self.view_size = int(action_size ** 0.5)  # 21
        assert self.view_size * self.view_size == action_size, \
            f"action_size must be a perfect square, got {action_size}"

        # Separate representation transforms (Level 3: decouple value/policy)
        self.policy_transform = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.LayerNorm(hidden_state_dim),
            nn.GELU(),
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.LayerNorm(hidden_state_dim),
            nn.GELU(),
        )
        self.value_transform = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.LayerNorm(hidden_state_dim),
            nn.GELU(),
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.LayerNorm(hidden_state_dim),
            nn.GELU(),
        )

        # Policy backbone (layers 0-5 match original checkpoint keys)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_state_dim, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
        )
        # Residual block for extra capacity
        self.policy_residual = nn.Sequential(
            nn.Linear(fc_hidden, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
        )
        # Factored output: row + col logits instead of flat 441
        self.policy_row = nn.Linear(fc_hidden, self.view_size)   # 512 → 21
        self.policy_col = nn.Linear(fc_hidden, self.view_size)   # 512 → 21

        self.support_size = 21 # Support set bins for values from -1 to 1 (-1.0, -0.9, ... 1.0)
        self.register_buffer('value_support', torch.linspace(-1.0, 1.0, self.support_size))
        self.value_net = nn.Sequential(
            nn.Linear(hidden_state_dim, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
            nn.Linear(fc_hidden, 3 * self.support_size),  # Categorical logits for 3 components
        )
        
        # Auxiliary Heads
        self.aux_heads = AuxiliaryHeads(hidden_state_dim, action_size)

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = hidden_state.shape[0]

        # Separate representation transforms (residual connection)
        policy_repr = hidden_state + self.policy_transform(hidden_state)
        value_repr = hidden_state + self.value_transform(hidden_state)

        # Policy path
        x = self.policy_net(policy_repr)
        x = x + self.policy_residual(x)

        # Factored policy: outer sum in log-space
        row_logits = self.policy_row(x)   # (B, 21)
        col_logits = self.policy_col(x)   # (B, 21)
        policy_logits = row_logits.unsqueeze(2) + col_logits.unsqueeze(1)  # (B, 21, 21)
        policy_logits = policy_logits.reshape(B, -1)  # (B, 441)
        
        # Logit clipping to prevent FP16 exponential overflow and NaNs before sequence/softmax
        policy_logits = torch.clamp(policy_logits, min=-10.0, max=10.0)

        # Value path using Support Set
        value_logits = self.value_net(value_repr)
        value_logits = value_logits.view(B, 3, self.support_size)
        value_probs = F.softmax(value_logits, dim=-1)
        value = (value_probs * self.value_support).sum(dim=-1) # (B, 3)
        
        # Aux path (use raw hidden state or policy repr? Use raw to force representation to encode it)
        threat_logits, opp_action_logits, heatmap_logits = self.aux_heads(hidden_state)
        
        return policy_logits, value, threat_logits, opp_action_logits, heatmap_logits

class AuxiliaryHeads(nn.Module):
    """
    Auxiliary prediction heads for enhanced representation learning.
    1. Threat Detection: (B, 3) logits [5, 6, 7]
    2. Opponent Action: (B, 441) logits (next player's move)
    3. Board Heatmap: (B, 1, 21, 21) logits (future 20 moves)
    """
    def __init__(self, hidden_state_dim: int, action_size: int = 441):
        super().__init__()
        self.view_size = int(action_size ** 0.5)
        
        # Threat Head
        self.threat_net = nn.Sequential(
            nn.Linear(hidden_state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 3) # Multi-label BCE targets
        )
        
        # Opponent Action Head (similar to Policy)
        self.opp_action_net = nn.Sequential(
            nn.Linear(hidden_state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, action_size) 
        )
        
        # Board Heatmap Head
        # Predict 21x21 occupancy map
        self.heatmap_net = nn.Sequential(
            nn.Linear(hidden_state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.view_size * self.view_size)
        )

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        threat_logits = self.threat_net(hidden_state)
        opp_action_logits = self.opp_action_net(hidden_state)
        heatmap_logits = self.heatmap_net(hidden_state) 
        # Heatmap reshape to (B, 1, 21, 21) if needed, but flat is fine for BCEWithLogitsLoss
        heatmap_logits = heatmap_logits.view(-1, 1, self.view_size, self.view_size)
        
        return threat_logits, opp_action_logits, heatmap_logits


class FocusNetwork(nn.Module):
    """
    Predicts the optimal view center (normalized r, c) from the global board state.
    Input: (B, 4, 100, 100)
    Output: (B, 2) values in [0, 1]
    Uses Spatial Softmax instead of FC+Sigmoid for enhanced numerical stability.
    """
    def __init__(self, channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, padding=2, stride=2), # 50x50
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), # 25x25
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), # 13x13
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1) # 13x13 heat map
        )
        
        # Create normalized coordinate grid for 13x13 (removed static buffer)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heatmap = self.conv(x) # (B, 1, H, W)
        B, C, H, W = heatmap.shape
        
        heatmap_flat = heatmap.reshape(B, -1) # (B, H*W)
        
        # Stable Softmax along spatial dimensions
        orig_dtype = heatmap_flat.dtype
        prob = F.softmax(heatmap_flat.float(), dim=-1).to(orig_dtype)
        
        # Dynamically generate coordinate grids matching the downsampled spatial dims
        coords_y = (torch.arange(H, dtype=torch.float32, device=x.device) + 0.5) / H
        coords_x = (torch.arange(W, dtype=torch.float32, device=x.device) + 0.5) / W
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
        
        grid_y = grid_y.reshape(1, -1) # (1, H*W)
        grid_x = grid_x.reshape(1, -1) # (1, H*W)
        
        # Compute expected coordinates (r=y, c=x)
        expected_y = (prob * grid_y).sum(dim=1, keepdim=True)
        expected_x = (prob * grid_x).sum(dim=1, keepdim=True)
        
        return torch.cat([expected_y, expected_x], dim=1)


class MuZeroNetwork(nn.Module):
    """
    Combined Gumbel MuZero network with:
    - DeepSeek MLA Transformer Backbone
    - Engram Episodic Memory
    - EfficientZero Consistency Head
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Representation: Transformer Backbone
        self.representation = TransformerBackbone(
            obs_channels=config.observation_channels,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_kv_compress=config.d_kv_compress,
            ffn_hidden=config.ffn_hidden,
            hidden_state_dim=config.hidden_state_dim,
            view_size=config.local_view_size,
            patch_size=config.patch_size,
            dropout=config.dropout
        )

        # 2. Engram Memory
        if config.use_engram:
            self.engram_module = EngramModule(
                hidden_dim=config.hidden_state_dim,
                value_dim=config.hidden_state_dim, # We simplify: value = hidden_state
                n_heads=config.memory_heads,
                top_k=config.memory_top_k
            )
        else:
            self.engram_module = None

        # 3. Consistency Head
        if config.use_consistency:
            self.consistency = ConsistencyModule(
                hidden_state_dim=config.hidden_state_dim,
                proj_dim=config.consistency_proj_dim
            )
        else:
            self.consistency = None
            
        # 4. State Reconstruction Head (Decoder)
        # Reconstructs the (8, 21, 21) local observation from the hidden state.
        # This acts as a strong self-supervised regularization.
        self.state_decoder = nn.Sequential(
            nn.Linear(config.hidden_state_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 8 * config.local_view_size * config.local_view_size)
        )

        # 4. Dynamics & Prediction
        self.dynamics = DynamicsNetwork(config.hidden_state_dim, config.policy_size, config.fc_hidden)
        self.prediction = PredictionNetwork(config.hidden_state_dim, config.policy_size, config.fc_hidden)
        
        # 5. Focus Network (Vision Attention)
        self.focus_net = FocusNetwork(channels=4)

        # 6. Session Context Encoder (Phase 4)
        # 4-dim input: [my_score_norm, opp1_score_norm, opp2_score_norm, games_remaining_norm]
        # Projects to hidden_state_dim for residual addition
        self.context_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, config.hidden_state_dim),
        )

        self.action_size = config.policy_size
        self.hidden_state_dim = config.hidden_state_dim

    def apply_session_context(self, hidden_state: torch.Tensor,
                              session_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply session context encoding to hidden state via residual addition.

        Args:
            hidden_state: (B, hidden_dim) tensor
            session_context: (B, 4) tensor or None. When None, returns hidden_state unchanged.

        Returns:
            Context-augmented hidden state, normalized to unit sphere.
        """
        if session_context is not None:
            ctx = self.context_encoder(session_context)  # (B, hidden_dim)
            hidden_state = hidden_state + ctx  # residual addition
            hidden_state = hidden_state / (hidden_state.norm(dim=1, keepdim=True) + 1e-8)
        return hidden_state

    def initial_inference(self, obs: torch.Tensor,
                          memory_keys: Optional[torch.Tensor] = None,
                          memory_values: Optional[torch.Tensor] = None,
                          session_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initial step: observation → (hidden_state, policy_logits, value).
        Incorporates memory read and session context if available.
        
        Args:
            obs: (B, C, H, W) observation tensor
            memory_keys: optional pre-retrieved memory keys
            memory_values: optional pre-retrieved memory values
            session_context: optional (B, 4) session context vector

        Returns:
            hidden_state, policy_logits, value
        """
        hidden_state = self.representation(obs)

        # Apply session context (Phase 4)
        hidden_state = self.apply_session_context(hidden_state, session_context)

        # Read from memory if provided
        # Read from memory if provided
        if self.engram_module is not None and memory_keys is not None and memory_values is not None:
            hidden_state = self.engram_module(hidden_state, memory_keys, memory_values)

        policy_logits, value, _, _, _ = self.prediction(hidden_state)
        if getattr(self.config, 'clip_value_reward', True):
            value = value.clamp(-10.0, 10.0)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state: torch.Tensor,
                            action: torch.Tensor,
                            memory_keys: Optional[torch.Tensor] = None,
                            memory_values: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recurrent step: (state, action) → (next_state, reward, policy, value).
        Incorporates memory read for the NEXT state.
        """
        # action is (B,) integer → one-hot
        action_onehot = F.one_hot(action.long(), self.action_size).float()
        next_state, reward = self.dynamics(hidden_state, action_onehot)
        if getattr(self.config, 'clip_value_reward', True):
            reward = reward.clamp(-5.0, 5.0)
        # Read from memory for next state
        if self.engram_module is not None and memory_keys is not None and memory_values is not None:
             next_state = self.engram_module(next_state, memory_keys, memory_values)

        policy_logits, value, _, _, _ = self.prediction(next_state)
        if getattr(self.config, 'clip_value_reward', True):
            value = value.clamp(-10.0, 10.0)
        return next_state, reward, policy_logits, value

    def project(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden state for consistency loss."""
        if self.consistency:
            return self.consistency.projector(hidden_state)
        return hidden_state # fallback (shouldn't happen if config correct)

    def reconstruct_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Reconstruct the local observation from the hidden state."""
        B = hidden_state.shape[0]
        flat_obs = self.state_decoder(hidden_state)
        vs = self.config.local_view_size
        return flat_obs.view(B, 8, vs, vs)

    def predict_projection(self, projected_state: torch.Tensor) -> torch.Tensor:
        """Predict expected projection (SimSiam predictor)."""
        if self.consistency:
            return self.consistency.predictor(projected_state)
        return projected_state

    @classmethod
    def from_config(cls, config) -> 'MuZeroNetwork':
        """Create network from MuZeroConfig."""
        return cls(config)

    def predict_center(self, global_state: torch.Tensor) -> Tuple[int, int]:
        """
        Predict the best center (row, col) from global state.
        global_state: (1, 4, 100, 100) tensor
        """
        # Ensure eval mode if not training
        training = self.training
        self.eval()
        
        with torch.no_grad():
            # (1, 2) in [0, 1]
            coords = self.focus_net(global_state) 
            r_norm = coords[0, 0].item()
            c_norm = coords[0, 1].item()
            
            # Convert to pixel coordinates
            w = global_state.shape[-1] # 100
            r = int(r_norm * w)
            c = int(c_norm * w)
            
            # Clamp
            r = max(0, min(w-1, r))
            c = max(0, min(w-1, c))
            
        if training:
            self.train()
            
        return r, c
