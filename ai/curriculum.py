"""Curriculum learning: per-stage board size and win length with graduation by games, win rate, loss, and optional Elo.

Maintainability: graduation thresholds and elasticity (curriculum_max_games, base_games) — see ai/MAINTENANCE.md.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ai.log_utils import get_logger

_log = get_logger(__name__)


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Coerce to float; return default on NaN/Inf or invalid."""
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default

@dataclass
class CurriculumStage:
    stage_id: int
    board_size: int
    win_length: int
    min_games: int = 0 # Calculated dynamically
    win_rate_threshold: float = 0.55 
    
    # New thresholds
    loss_threshold: float = 0.005 # Max absolute derivative for convergence
    elo_threshold: float = 0.0    # Min Elo gain vs baseline

class CurriculumManager:
    """
    Per-stage graduation: min_games (scaled by board size), Wilson LCB, loss
    convergence, optional delta-Elo. Elasticity: curriculum_max_games caps
    per-stage min_games; curriculum_base_games sets the 15x15 baseline.
    """
    def __init__(self, config):
        self.config = config

        # Base settings (15x15 = base_games; larger boards scale by (S_k/S_1)^alpha)
        self.base_games = getattr(config, 'curriculum_base_games', 10000)
        self.base_size = 15 * 15  # 225
        self.scaling_alpha = 1.5
        
        # Initialize Stages with Dynamic Game Counts
        self.stages = [
            self._create_stage(1, 15, 5, wr=0.60),
            self._create_stage(2, 30, 6, wr=0.58), 
            self._create_stage(3, 50, 7, wr=0.55),
            self._create_stage(4, 100, 8, wr=0.55)
        ]
        
        self.current_stage_idx = 0
        self.games_in_stage = 0
        
        # Stats Buffers
        self.win_rate_buffer = []   # Per-game result: 1.0 win, 0.5 draw, 0.0 loss
        self.loss_buffer = []       # For convergence check (raw or EMA)
        self._loss_ema = None       # Optional EMA of loss
        self.loss_ema_buffer = []   # Used for convergence when use_ema=True
        self.use_ema_loss = getattr(config, 'curriculum_use_ema_loss', True)
        self.ema_alpha = getattr(config, 'curriculum_ema_alpha', 0.1)
        
        # Elo baseline for this stage (set on advance); used for delta-Elo graduation
        self.stage_start_elo = None
        # LCB must exceed this to ensure above random (e.g. 0.50)
        self.baseline_wr = getattr(config, 'curriculum_baseline_wr', 0.50)

        # Probation: regression protection after graduation
        self.in_probation = False
        self.probation_win_buffer: List[float] = []
        self.probation_games = getattr(config, 'curriculum_probation_games', 500)
        self.probation_wr_floor = getattr(config, 'curriculum_probation_wr_floor', 0.40)
        self._pre_probation_state: Optional[Dict[str, Any]] = None
        
    def _create_stage(self, stage_id, size, win_len, wr):
        # Dynamic Game Count Formula: N_k = N_1 * (S_k / S_1)^alpha
        s_k = size * size
        ratio = s_k / self.base_size
        min_games = int(self.base_games * (ratio ** self.scaling_alpha))
        
        # Cap to avoid runaway size for 100x100, but must be >= 50x50 count (monotonic)
        # 50x50 ~ 37 * base_games; 100x100 ~ 296 * base_games. Cap so 100x100 >= 50x50.
        cap = getattr(self.config, 'curriculum_max_games', 2_000_000)
        if min_games > cap:
            min_games = cap
        
        elo_delta = getattr(self.config, 'curriculum_elo_threshold', 35.0)
        return CurriculumStage(
            stage_id=stage_id,
            board_size=size,
            win_length=win_len,
            min_games=min_games,
            win_rate_threshold=wr,
            elo_threshold=elo_delta
        )

    def set_stage(self, stage_idx: int):
        if 0 <= stage_idx < len(self.stages):
            self.current_stage_idx = stage_idx
            self.games_in_stage = 0
            self.win_rate_buffer = []
            self.loss_buffer = []
            self.loss_ema_buffer = []
            self._loss_ema = None
            self.stage_start_elo = None
            _log.info("Manually set to Stage %d: %s", stage_idx + 1, self.get_current_stage())

    def get_current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]
        
    def record_game_result(self, score: float):
        """
        Record one game result for graduation stats. Call only when a game finishes.
        score: 1.0 win, 0.5 draw, 0.0 loss (for the current/active agent).
        """
        s = _safe_float(score, default=-1.0)
        if s < 0 or s > 1:
            return  # skip invalid; optional: log once
        self.games_in_stage += 1
        self.win_rate_buffer.append(s)
        if len(self.win_rate_buffer) > 1000:
            self.win_rate_buffer.pop(0)

        # Probation check: track win rate after graduation
        if self.in_probation:
            self.probation_win_buffer.append(s)
            if len(self.probation_win_buffer) >= self.probation_games:
                avg = np.mean(self.probation_win_buffer)
                if avg < self.probation_wr_floor:
                    _log.warning(
                        "Probation FAILED: avg WR %.3f < floor %.3f after %d games. Reverting to previous stage.",
                        avg, self.probation_wr_floor, len(self.probation_win_buffer)
                    )
                    self._revert_from_probation()
                else:
                    _log.info(
                        "Probation PASSED: avg WR %.3f >= floor %.3f. Stage confirmed.",
                        avg, self.probation_wr_floor
                    )
                    self.in_probation = False
                    self.probation_win_buffer = []
                    self._pre_probation_state = None

    def record_loss(self, loss: float):
        """
        Record training loss for convergence check. Does not touch games_in_stage or win_rate_buffer.
        Invalid/NaN/Inf loss is skipped.
        """
        v = _safe_float(loss, default=float('nan'))
        if math.isnan(v):
            return
        self.loss_buffer.append(v)
        if len(self.loss_buffer) > 2000:
            self.loss_buffer.pop(0)
        if self.use_ema_loss:
            self._loss_ema = v if self._loss_ema is None else (
                (1.0 - self.ema_alpha) * self._loss_ema + self.ema_alpha * v
            )
            self.loss_ema_buffer.append(self._loss_ema)
            if len(self.loss_ema_buffer) > 2000:
                self.loss_ema_buffer.pop(0)

    def update_stats(self, win_rate_vs_frozen: float, loss: float):
        """
        Legacy: update both game result and loss. Prefer record_game_result + record_loss separately.
        """
        self.record_game_result(win_rate_vs_frozen)
        self.record_loss(loss)
            
    def _check_loss_convergence(self, window=500, epsilon=1e-4) -> bool:
        """
        Check if loss has plateaued using first derivative approximation.
        | 1/W * sum(L_t - L_{t-W}) | < epsilon
        Uses loss_ema_buffer when use_ema_loss is True, else loss_buffer.
        """
        buf = self.loss_ema_buffer if (self.use_ema_loss and len(self.loss_ema_buffer) >= window * 2) else self.loss_buffer
        if len(buf) < window * 2:
            return False
        current_loss = np.mean(buf[-window:])
        prev_loss = np.mean(buf[-2*window:-window])
        derivative = abs(current_loss - prev_loss)
        return derivative < epsilon

    def _wilson_score_lower_bound(self, positive, n, confidence=0.99) -> float:
        """
        Calculate Wilson Score Interval Lower Bound.
        confidence 0.99 -> z approx 2.576
        Returns 0.0 if n <= 0 or positive not in [0, n].
        """
        try:
            n = int(n)
            positive = float(positive)
        except (TypeError, ValueError):
            return 0.0
        if n <= 0 or positive < 0 or positive > n:
            return 0.0
        phat = positive / n
        z = 2.576  # 99% confidence
        numerator = phat + z*z/(2*n) - z * np.sqrt((phat*(1-phat) + z*z/(4*n))/n)
        denominator = 1 + z*z/n
        return numerator / denominator

    def check_graduation(self, league=None) -> bool:
        """
        Check if we should advance to the next stage.
        Criteria:
        1. Played minimum games (Dynamic).
        2. Wilson Lower Bound of Win Rate > Threshold and > baseline (e.g. 0.50).
        3. Loss Convergence (derivative < epsilon).
        4. Optional: Delta Elo vs stage baseline >= stage.elo_threshold (if league and stage_start_elo set).
        """
        stage = self.get_current_stage()
        
        # 1. Min Games
        if self.games_in_stage < stage.min_games:
            if self.games_in_stage >= max(0, stage.min_games - 2000):
                _log.info("Not graduating: games %d < min_games %d", self.games_in_stage, stage.min_games)
            return False

        # 2. Max Stage check
        if self.current_stage_idx >= len(self.stages) - 1:
            return False

        # 3. Wilson Score Check (Statistical Significance)
        if not self.win_rate_buffer:
            _log.info("Not graduating: win_rate_buffer empty")
            return False

        n_samples = len(self.win_rate_buffer)
        avg_wr = np.mean(self.win_rate_buffer)
        wins = np.sum(self.win_rate_buffer)

        lcb = self._wilson_score_lower_bound(wins, n_samples)

        if lcb < stage.win_rate_threshold:
            _log.info("Not graduating: LCB %.3f < threshold %.3f (avg_wr=%.3f)", lcb, stage.win_rate_threshold, avg_wr)
            return False
        if lcb < self.baseline_wr:
            _log.info("Not graduating: LCB %.3f < baseline %.2f", lcb, self.baseline_wr)
            return False

        # 4. Loss Convergence Check
        if not self._check_loss_convergence():
            _log.info("Not graduating: loss not converged (need derivative < epsilon)")
            return False

        # 5. Elo condition (optional)
        current_elo = getattr(league, 'current_elo', None) if league is not None else None
        if league is not None and self.stage_start_elo is not None and getattr(stage, 'elo_threshold', 0) > 0:
            if current_elo is not None:
                delta_elo = current_elo - self.stage_start_elo
                if delta_elo < stage.elo_threshold:
                    _log.info("Not graduating: delta_elo %.1f < %s", delta_elo, stage.elo_threshold)
                    return False
            
        # If all pass:
        _log.info("Graduation triggered: Stage %s complete. Games=%d, LCB=%.3f, loss converged.", stage.stage_id, self.games_in_stage, lcb)
        if current_elo is not None and self.stage_start_elo is not None:
            _log.info("Delta Elo: %.1f >= %s", current_elo - self.stage_start_elo, stage.elo_threshold)
        
        return True
        
    def advance(self, league=None):
        """Advance to next stage with probation. Saves pre-advance state for potential rollback."""
        if self.current_stage_idx < len(self.stages) - 1:
            # Save state for rollback if probation fails
            self._pre_probation_state = self.state_dict()
            
            self.current_stage_idx += 1
            self.games_in_stage = 0
            self.win_rate_buffer = []
            self.loss_buffer = []
            self.loss_ema_buffer = []
            self._loss_ema = None
            if league is not None:
                self.stage_start_elo = getattr(league, 'current_elo', None)
            
            # Enter probation
            self.in_probation = True
            self.probation_win_buffer = []
            _log.info("Advanced to Stage %d. Entering probation (%d games, floor=%.2f).",
                      self.current_stage_idx + 1, self.probation_games, self.probation_wr_floor)
            return self.get_current_stage()
        return None

    def _revert_from_probation(self):
        """Revert to pre-graduation state when probation fails."""
        if self._pre_probation_state is not None:
            self.load_state_dict(self._pre_probation_state)
            self._pre_probation_state = None
            self.in_probation = False
            self.probation_win_buffer = []
            _log.info("Reverted to Stage %d after probation failure.", self.current_stage_idx + 1)
        
    def state_dict(self) -> Dict[str, Any]:
        """Return the current state of the curriculum."""
        return {
            'current_stage_idx': self.current_stage_idx,
            'games_in_stage': self.games_in_stage,
            'win_rate_buffer': self.win_rate_buffer,
            'loss_buffer': self.loss_buffer,
            'stage_start_elo': self.stage_start_elo,
            'loss_ema_buffer': getattr(self, 'loss_ema_buffer', []),
            '_loss_ema': getattr(self, '_loss_ema', None),
            'in_probation': self.in_probation,
            'probation_win_buffer': self.probation_win_buffer,
            '_pre_probation_state': self._pre_probation_state,
        }

    def _ensure_number_list(self, val: Any, name: str, max_len: int = 5000) -> List[float]:
        """Return a list of finite numbers; else empty list (with optional warning)."""
        if not isinstance(val, list):
            return []
        out = []
        for x in val:
            try:
                f = float(x)
                if math.isfinite(f):
                    out.append(f)
            except (TypeError, ValueError):
                continue
        if len(out) != len(val) and val:
            _log.warning("%s had invalid entries, using %d valid.", name, len(out))
        return out[:max_len]

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore curriculum state. Backward compatible with older checkpoints missing new keys."""
        self.current_stage_idx = int(state_dict.get('current_stage_idx', 0))
        self.games_in_stage = int(state_dict.get('games_in_stage', 0))
        self.win_rate_buffer = self._ensure_number_list(
            state_dict.get('win_rate_buffer'), 'win_rate_buffer', max_len=1000
        )
        self.loss_buffer = self._ensure_number_list(
            state_dict.get('loss_buffer'), 'loss_buffer', max_len=2000
        )
        self.stage_start_elo = state_dict.get('stage_start_elo', None)
        self.loss_ema_buffer = self._ensure_number_list(
            state_dict.get('loss_ema_buffer'), 'loss_ema_buffer', max_len=2000
        )
        self._loss_ema = state_dict.get('_loss_ema', None)
        if self._loss_ema is not None:
            try:
                self._loss_ema = float(self._loss_ema)
                if not math.isfinite(self._loss_ema):
                    self._loss_ema = None
            except (TypeError, ValueError):
                self._loss_ema = None
        # Probation state
        self.in_probation = bool(state_dict.get('in_probation', False))
        self.probation_win_buffer = self._ensure_number_list(
            state_dict.get('probation_win_buffer'), 'probation_win_buffer', max_len=1000
        )
        self._pre_probation_state = state_dict.get('_pre_probation_state', None)
        self.current_stage_idx = max(0, min(self.current_stage_idx, len(self.stages) - 1))
        _log.info("State loaded: Stage %d, Games %d, Probation=%s",
                  self.current_stage_idx + 1, self.games_in_stage, self.in_probation)
