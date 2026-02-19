"""Board data augmentation for more training diversity.

Available methods:
1. Rotation + mirror (default): 0/90/180/270° and horizontal flip → 8 symmetries.
2. Observation noise (optional): additive Gaussian noise on observations for robustness.

Not used here:
- Player/color permutation: state is already from current player's perspective (me/next/prev),
  so permuting 1/2/3 does not change the batch.
- Vertical mirror: already covered by rotation + horizontal mirror.
"""

import numpy as np
from typing import Dict, Tuple, Optional

from ai.log_utils import get_logger

_log = get_logger(__name__)

VIEW_SIZE = 21


def _rot90_ccw(r: int, c: int, k: int, size: int) -> Tuple[int, int]:
    """Apply k*90° counter-clockwise rotation. (r,c) in [0, size-1]."""
    if k == 0:
        return r, c
    if k == 1:
        return c, size - 1 - r
    if k == 2:
        return size - 1 - r, size - 1 - c
    if k == 3:
        return size - 1 - c, r
    return r, c


def _inv_rot90_ccw(r: int, c: int, k: int, size: int) -> Tuple[int, int]:
    """Inverse of _rot90_ccw (so we map new view back to old view)."""
    if k == 0:
        return r, c
    if k == 1:
        return size - 1 - c, r
    if k == 2:
        return size - 1 - r, size - 1 - c
    if k == 3:
        return c, size - 1 - r
    return r, c


def _action_inv_perm_21(rot: int, mirror: int) -> np.ndarray:
    """
    For 21x21 local action space: after we apply (rot, mirror) to the view,
    new_policy[new_idx] = old_policy[old_idx] where (old_r, old_c) maps to (new_r, new_c).
    Returns inv_perm[441] so that new_policy = old_policy[inv_perm].
    """
    inv_perm = np.zeros(VIEW_SIZE * VIEW_SIZE, dtype=np.int32)
    for new_r in range(VIEW_SIZE):
        for new_c in range(VIEW_SIZE):
            r1, c1 = new_r, VIEW_SIZE - 1 - new_c if mirror else new_c
            old_r, old_c = _inv_rot90_ccw(r1, c1, rot, VIEW_SIZE)
            old_idx = old_r * VIEW_SIZE + old_c
            new_idx = new_r * VIEW_SIZE + new_c
            inv_perm[new_idx] = old_idx
    return inv_perm


def _action_fwd_perm_21(rot: int, mirror: int) -> np.ndarray:
    """Forward perm: old_idx -> new_idx. new_action = fwd_perm[old_action]."""
    fwd = np.zeros(VIEW_SIZE * VIEW_SIZE, dtype=np.int32)
    for old_r in range(VIEW_SIZE):
        for old_c in range(VIEW_SIZE):
            new_r, new_c = _rot90_ccw(old_r, old_c, rot, VIEW_SIZE)
            if mirror:
                new_c = VIEW_SIZE - 1 - new_c
            old_idx = old_r * VIEW_SIZE + old_c
            new_idx = new_r * VIEW_SIZE + new_c
            fwd[old_idx] = new_idx
    return fwd


def _center_transform(row: int, col: int, board_size: int, rot: int, mirror: int) -> Tuple[int, int]:
    """Transform board (row, col) by rot then mirror. Returns (new_row, new_col)."""
    r, c = _rot90_ccw(row, col, rot, board_size)
    if mirror:
        c = board_size - 1 - c
    return r, c


def _transform_spatial_nd(x: np.ndarray, rot: int, mirror: int) -> np.ndarray:
    """Apply rot (0..3) then horizontal flip to last two dimensions. x: (..., H, W)."""
    if rot != 0:
        x = np.rot90(x, rot, axes=(-2, -1)).copy()
    if mirror:
        x = np.flip(x, axis=-1).copy()
    return x


def apply_observation_noise(
    batch: Dict[str, np.ndarray], rng: np.random.Generator, std: float = 0.02
) -> None:
    """Add Gaussian noise to observations and next_observations. Clips to [0, 1]. In-place."""
    if std <= 0:
        return
    if batch is None or not isinstance(batch, dict):
        raise ValueError("batch must be a non-empty dict")
    obs = batch.get('observations')
    if obs is None or not isinstance(obs, np.ndarray) or obs.size == 0:
        raise ValueError("batch['observations'] must be a non-empty numpy array")
    B = obs.shape[0]
    for b in range(B):
        noise = rng.normal(0, std, batch['observations'][b].shape).astype(np.float32)
        batch['observations'][b] = np.clip(batch['observations'][b] + noise, 0.0, 1.0)
        noise_next = rng.normal(0, std, batch['next_observations'][b].shape).astype(np.float32)
        batch['next_observations'][b] = np.clip(batch['next_observations'][b] + noise_next, 0.0, 1.0)


def apply_board_augment(
    batch: Dict[str, np.ndarray],
    rng: np.random.Generator,
    noise_std: Optional[float] = None,
) -> None:
    """
    Apply random rotation (0/90/180/270°) and horizontal mirror to each sample in the batch.
    Optionally add Gaussian noise to observations (noise_std > 0).
    Modifies batch in place. Uses VIEW_SIZE=21 for actions and batch['global_states'].shape[-1] for board.
    """
    if batch is None or not isinstance(batch, dict):
        raise ValueError("batch must be a non-empty dict")
    obs = batch.get('observations')
    if obs is None or not isinstance(obs, np.ndarray) or obs.size == 0:
        raise ValueError("batch['observations'] must be a non-empty numpy array")
    gs = batch.get('global_states')
    if gs is None or not isinstance(gs, np.ndarray) or gs.size == 0:
        raise ValueError("batch['global_states'] must be a non-empty numpy array")
    B = obs.shape[0]
    view_size = VIEW_SIZE
    board_size = gs.shape[-1]
    action_size = view_size * view_size

    for b in range(B):
        rot = int(rng.integers(0, 4))
        mirror = int(rng.integers(0, 2))
        if rot == 0 and mirror == 0:
            continue

        inv_perm = _action_inv_perm_21(rot, mirror)
        fwd_perm = _action_fwd_perm_21(rot, mirror)

        # Observations (B, C, H, W) and next_observations
        batch['observations'][b] = _transform_spatial_nd(batch['observations'][b], rot, mirror)
        batch['next_observations'][b] = _transform_spatial_nd(batch['next_observations'][b], rot, mirror)
        # Global states (B, 4, board_size, board_size)
        batch['global_states'][b] = _transform_spatial_nd(batch['global_states'][b], rot, mirror)

        # Actions (B, K): old index -> new index via fwd_perm
        K = batch['actions'].shape[1]
        for k in range(K):
            a = int(batch['actions'][b, k])
            if 0 <= a < action_size:
                batch['actions'][b, k] = fwd_perm[a]

        # Target policies (B, K+1, action_size): new_policy = old_policy[inv_perm]
        for k in range(batch['target_policies'].shape[1]):
            old_p = batch['target_policies'][b, k]
            batch['target_policies'][b, k] = old_p[inv_perm]

        # Target center: single int (row*board_size+col)
        flat = int(batch['target_centers'][b])
        row, col = flat // board_size, flat % board_size
        nr, nc = _center_transform(row, col, board_size, rot, mirror)
        batch['target_centers'][b] = nr * board_size + nc

        # Target opponent actions: old index -> new index
        oa = int(batch['target_opponent_actions'][b])
        if 0 <= oa < action_size:
            batch['target_opponent_actions'][b] = fwd_perm[oa]
        # -1 for terminal stays -1

        # Target heatmaps (B, 21, 21)
        batch['target_heatmaps'][b] = _transform_spatial_nd(
            batch['target_heatmaps'][b][np.newaxis, ...], rot, mirror
        ).squeeze(0)

    if noise_std is not None and noise_std > 0:
        apply_observation_noise(batch, rng, noise_std)
    return None
