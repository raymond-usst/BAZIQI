"""Render game board to image for saving self-play samples (wooden board style)."""

import os
import numpy as np

from ai.log_utils import get_logger

_log = get_logger(__name__)

# Optional matplotlib for image export
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _wood_background(h: int, w: int, cell_px: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a wooden board background with light grain."""
    # Base wood color (warm tan)
    base = np.array([0.72, 0.53, 0.35], dtype=np.float32)
    # Slight per-pixel variation for grain (noise)
    noise = rng.uniform(-0.06, 0.06, (h * cell_px, w * cell_px, 3))
    wood = np.clip(base + noise, 0, 1).astype(np.float32)
    # Subtle horizontal grain bands
    y = np.arange(h * cell_px, dtype=np.float32) / max(1, cell_px * 4)
    grain = 0.03 * np.sin(y * 1.2)[:, np.newaxis] * np.ones((1, w * cell_px))
    wood[:, :, :] += np.clip(grain[:, :, np.newaxis], -0.05, 0.05)
    return np.clip(wood, 0, 1)


def board_to_image_path(board: np.ndarray, path: str, cell_px: int = 10) -> bool:
    """
    Render a board (H, W) with values 0=empty, 1=red, 2=green, 3=blue to PNG
    on a wooden board with grid lines and round pieces.

    Args:
        board: int8 (H, W), 0=empty, 1/2/3=players
        path: output file path (e.g. img/game_100.png)
        cell_px: size of each cell in pixels (default 10; 100x100 board -> 1000x1000 image)

    Returns:
        True if saved, False if matplotlib not available or error.
    """
    if not _HAS_MPL:
        return False
    if board is None or not isinstance(board, np.ndarray) or board.size == 0:
        raise ValueError("board must be a non-empty numpy array")
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        h, w = board.shape
        rng = np.random.default_rng(42)
        # 1) Wood background
        wood = _wood_background(h, w, cell_px, rng)
        dpi = 100
        fig, ax = plt.subplots(1, 1, figsize=(w * cell_px / dpi, h * cell_px / dpi), dpi=dpi)
        ax.imshow(wood, origin='upper', interpolation='bilinear', extent=[0, w, h, 0])
        # 2) Grid lines (dark wood / ink)
        line_color = (0.35, 0.22, 0.12)
        for i in range(w + 1):
            ax.axvline(i, 0, h, color=line_color, lw=0.5)
        for i in range(h + 1):
            ax.axhline(i, 0, w, color=line_color, lw=0.5)
        # Margin so edge pieces (radius 0.42) are not cropped; show full board
        margin = 0.55
        ax.set_xlim(-margin, w - 1 + margin)
        ax.set_ylim(h - 1 + margin, -margin)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        # 3) Pieces as circles at grid line intersections (not cell centers)
        piece_colors = {
            1: (0.85, 0.25, 0.22),
            2: (0.22, 0.65, 0.28),
            3: (0.22, 0.42, 0.82),
        }
        radius = 0.42  # slightly smaller than half spacing (1) so pieces don't overlap
        for r in range(h):
            for c in range(w):
                pid = int(board[r, c])
                if pid not in piece_colors:
                    continue
                cx, cy = c, r
                circle = Circle((cx, cy), radius, color=piece_colors[pid], ec=(0.2, 0.2, 0.2), lw=0.35)
                ax.add_patch(circle)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        return True
    except Exception:
        return False
