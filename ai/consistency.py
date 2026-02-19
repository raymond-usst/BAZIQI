"""
EfficientZero self-supervised consistency loss module.

Enforces temporal consistency: the dynamics model's predicted next hidden state
should be consistent with the representation network's encoding of the actual
next observation. Uses SimSiam-style asymmetric architecture to prevent collapse.

Reference: "Mastering Atari Games with Limited Data" (Ye et al., 2021)
"""

import torch
import torch.nn as nn

from ai.log_utils import get_logger

_log = get_logger(__name__)


class ConsistencyModule(nn.Module):
    """
    Self-supervised consistency module with projector + predictor (SimSiam-style).

    Loss: L = 2 - 2*cos_sim(predict(project(h_pred)), sg(project(h_actual))) after normalize.
    Equivalently: per-sample loss = 2.0 - 2.0 * (p_pred * z_actual).sum(dim=-1) for unit vectors.

    Where:
        h_predicted = dynamics(h_t, a_t)        — predicted next state
        h_actual = representation(obs_{t+1})     — actual next state
        sg() = stop-gradient
    """

    def __init__(self, hidden_state_dim: int = 128, proj_dim: int = 256):
        super().__init__()
        # Projector: hidden_state → projection space
        self.projector = nn.Sequential(
            nn.Linear(hidden_state_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

        # Predictor: asymmetric head to prevent collapse (SimSiam)
        pred_hidden = proj_dim // 2
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )

    def forward(self, h_predicted: torch.Tensor,
                h_actual: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Compute consistency loss between predicted and actual hidden states.

        Args:
            h_predicted: (B, hidden_dim) — dynamics model output (gradient flows)
            h_actual: (B, hidden_dim) — representation network output (stop-gradient)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss: consistency loss
        """
        # Project predicted side (gradient flows, BN updates running stats)
        z_pred = self.projector(h_predicted)

        # Project actual side in eval mode to prevent BN running stats contamination.
        self.projector.eval()
        with torch.no_grad():
            z_actual = self.projector(h_actual)
        self.projector.train()

        # Predict from predicted side (asymmetric head prevents collapse)
        p_pred = self.predictor(z_pred)

        # Stop gradient on actual side (SimSiam)
        z_actual = z_actual.detach()

        # Normalize
        p_pred = nn.functional.normalize(p_pred, dim=-1)
        z_actual = nn.functional.normalize(z_actual, dim=-1)

        # Negative cosine similarity → per-sample loss
        per_sample_loss = 2.0 - 2.0 * (p_pred * z_actual).sum(dim=-1)

        if reduction == 'mean':
            return per_sample_loss.mean()
        elif reduction == 'sum':
            return per_sample_loss.sum()
        elif reduction == 'none':
            return per_sample_loss
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
