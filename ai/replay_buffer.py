"""Experience replay buffer for MuZero training.

Features:
- Memory-aware capacity management (configurable GB budget)
- Multi-factor quality scoring (outcome, length, recency, policy sharpness, model maturity)
- Tiered eviction (protects recent games, evicts lowest quality first)
- Prioritized experience replay (higher quality games sampled more frequently)
- System memory safety valve (emergency eviction when RAM critically low)
- Backward-compatible save/load with old format
"""

import numpy as np
import pickle
import os
import time
import threading
from typing import List, Tuple, Optional, Dict, Any

from ai.log_utils import get_logger

_log = get_logger(__name__)


class GameHistory:
    """Stores the trajectory of a single game."""

    SNAPSHOT_INTERVAL = 10  # Save board snapshot every N steps

    def __init__(self):
        self.observations: List[np.ndarray] = []       # local observations
        self.actions: List[int] = []                     # local action indices
        self.rewards: List[float] = []                   # immediate rewards
        self.policy_targets: List[np.ndarray] = []       # MCTS visit distributions
        self.root_values: List[np.ndarray] = []          # MCTS root values (vector)
        self.threats: List[np.ndarray] = []              # Threat vectors [5, 6, 7+]
        self.player_ids: List[int] = []                  # who played each move
        self.centers: List[Tuple[int, int]] = []         # crop centers
        self.done: bool = False
        self.winner: Optional[int] = None
        # Ranking info (Phase 2): [(pid, placement_0indexed)] sorted 1st→3rd
        self.rankings: List[Tuple[int, int]] = []
        # Per-player placement reward: {pid: reward_float}
        self.placement_rewards: Dict[int, float] = {}
        # Session context (Phase 3-4): raw session-level data for context encoding
        self.session_scores: Optional[Dict[int, int]] = None  # {pid: cumulative_points} at game start
        self.session_game_idx: int = 0           # which game in the session (0-indexed)
        self.session_length: int = 1             # total games in session
        # Board snapshots for fast reconstruction in sample_batch (Opt 5)
        # {step_index: board_copy_int8_100x100}
        self.board_snapshots: Dict[int, np.ndarray] = {}
        # Precomputed focus data (computed once at save_game time)
        # board_states[i] = int8(100,100) board BEFORE move i (perspective-agnostic)
        self.board_states: Optional[List[np.ndarray]] = None
        # target_centers_precomputed[i] = int index (row*100+col) of the actual move at step i
        self.target_centers_precomputed: Optional[List[int]] = None
        
        # Phase 3: Curriculum Learning Support
        self.board_size: int = 100
        self.win_length: int = 8
        # Final board state (for visualization); set at end of self-play
        self.final_board: Optional[np.ndarray] = None

    def store(self, observation: np.ndarray, action: int, reward: float,
              policy: np.ndarray, root_value: np.ndarray, threats: np.ndarray,
              player_id: int, center: Tuple[int, int]):
        """Append one step: observation, action, reward, policy target, root value, threats, player_id, center."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policy_targets.append(policy)
        self.root_values.append(root_value)
        self.threats.append(threats)
        self.player_ids.append(player_id)
        self.centers.append(center)

    def __len__(self):
        return len(self.observations)


class ReplayBuffer:
    """Memory-aware replay buffer with quality-based retention and prioritized sampling.

    Instead of a simple FIFO circular buffer, this implementation:
    1. Tracks estimated memory per game and enforces a configurable GB budget
    2. Scores each game's quality based on multiple factors
    3. When over budget, batch-evicts lowest-quality older games (protecting recent ones)
    4. Samples games proportional to quality^alpha (prioritized experience replay)
    5. Monitors system RAM as emergency safety valve

    Elasticity: Growth is bounded by max_size and max_memory_gb. Eviction runs when
    total_memory or len(games) exceeds budget. sample_batch() requires at least one
    game (raises ValueError with num_games=0 otherwise). Producers (e.g. learner
    receiving games) may block on full queue until consumer drains.
    """

    # ── Quality scoring weight factors ──
    W_OUTCOME   = 0.25   # Decisive win vs draw
    W_LENGTH    = 0.20   # Game length in optimal range
    W_RECENCY   = 0.25   # How recently the game was generated
    W_SHARPNESS = 0.15   # MCTS policy confidence (low entropy)
    W_MATURITY  = 0.15   # Training step of the model that generated the game

    # ── Length scoring parameters ──
    OPTIMAL_LEN_MIN = 30
    OPTIMAL_LEN_MAX = 120
    MIN_USEFUL_LEN  = 10
    MAX_USEFUL_LEN  = 250

    # ── Eviction parameters ──
    EVICTION_TARGET_RATIO  = 0.85  # Evict down to 85% of budget (hysteresis)
    RECENT_PROTECT_RATIO   = 0.20  # Protect newest 20% from eviction
    SYSTEM_RAM_EMERGENCY_GB = 3.0  # Emergency eviction if available RAM < this
    # Scalability: skip O(n) num_positions() in memory_report when game count exceeds this
    _REPORT_NUM_POSITIONS_MAX_GAMES = 5000

    def __init__(self, max_size: int = 50000, max_memory_gb: float = 35.0,
                 priority_alpha: float = 0.6, min_games: int = 100):
        """
        Args:
            max_size: Maximum number of games (hard cap regardless of memory)
            max_memory_gb: Memory budget in GB for the buffer contents
            priority_alpha: Sampling priority sharpness (0=uniform, 1=fully proportional)
            min_games: Minimum games to keep (won't evict below this)
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_gb * (1024 ** 3))
        self.priority_alpha = priority_alpha
        self.min_games = min_games

        # Storage
        self.games: List[GameHistory] = []
        self.meta: List[Dict[str, Any]] = []

        # Tracking
        self.total_memory = 0          # Current estimated memory usage (bytes)
        self.total_games = 0           # Monotonic counter of all games ever added
        self._eviction_count = 0       # Total games evicted across lifetime

        # Cached sampling weights (invalidated on any mutation)
        self._sampling_weights: Optional[np.ndarray] = None
        self._weights_dirty = True

        # Thread safety: protects self.games / self.meta against concurrent
        # save_game (main thread) and sample_batch (prefetch thread)
        self._data_lock = threading.Lock()

    def clear(self):
        """Reset the buffer to empty (for curriculum transitions)."""
        with self._data_lock:
            self.games = []
            self.meta = []
            self.total_memory = 0
            self._sampling_weights = None
            self._weights_dirty = True
            print("[ReplayBuffer] Buffer cleared.")

    # ================================================================
    #  Memory Estimation
    # ================================================================

    @staticmethod
    def _estimate_game_memory(game: GameHistory) -> int:
        """Estimate memory footprint of a GameHistory in bytes."""
        n = len(game)
        if n == 0:
            return 256  # Minimal empty-object overhead

        # Numpy array sizes (measured from actual data)
        obs_bytes = sum(o.nbytes for o in game.observations)
        policy_bytes = sum(p.nbytes for p in game.policy_targets)

        # Board snapshots (100x100 int8 = 10KB each)
        snapshots = getattr(game, 'board_snapshots', {})
        snapshot_bytes = sum(s.nbytes for s in snapshots.values()) if snapshots else 0

        # Precomputed board states (sparse dict or legacy per-position list)
        board_states = getattr(game, 'board_states', None)
        if board_states:
            if isinstance(board_states, dict):
                precomputed_bytes = sum(b.nbytes for b in board_states.values())
            else:
                precomputed_bytes = sum(b.nbytes for b in board_states)
        else:
            precomputed_bytes = 0

        # Python scalar objects per step:
        #   action(int~28) + reward(float~24) + root_value(float~24)
        #   + player_id(int~28) + center(tuple~72) + 7 list-ptrs(~56)
        scalar_bytes = n * 232

        # Base object overhead (GameHistory + lists + bool + Optional + dicts)
        base_overhead = 1024

        # Additional metadata fields (rankings, placement_rewards, session_scores, etc.)
        metadata_bytes = 256

        return obs_bytes + policy_bytes + snapshot_bytes + precomputed_bytes + scalar_bytes + metadata_bytes + base_overhead

    # ================================================================
    #  Quality Scoring
    # ================================================================

    def _compute_quality(self, game: GameHistory, meta: Dict) -> float:
        """Compute multi-factor quality score in [0, 1]."""
        score = 0.0
        game_len = len(game)

        # ── 1. Outcome: decisive wins are more informative ──
        outcome_s = 1.0 if game.winner is not None else 0.3
        score += self.W_OUTCOME * outcome_s

        # ── 2. Length: bell-curve around optimal range ──
        if game_len < self.MIN_USEFUL_LEN:
            length_s = 0.1
        elif game_len < self.OPTIMAL_LEN_MIN:
            t = (game_len - self.MIN_USEFUL_LEN) / max(1, self.OPTIMAL_LEN_MIN - self.MIN_USEFUL_LEN)
            length_s = 0.3 + 0.7 * t
        elif game_len <= self.OPTIMAL_LEN_MAX:
            length_s = 1.0
        elif game_len <= self.MAX_USEFUL_LEN:
            t = (game_len - self.OPTIMAL_LEN_MAX) / max(1, self.MAX_USEFUL_LEN - self.OPTIMAL_LEN_MAX)
            length_s = 1.0 - 0.6 * t
        else:
            length_s = 0.2
        score += self.W_LENGTH * length_s

        # ── 3. Recency: exponential decay (half-life ≈ 3500 games) ──
        age = max(0, self.total_games - meta.get('insert_idx', 0))
        recency_s = np.exp(-age / 5000.0)
        score += self.W_RECENCY * recency_s

        # ── 4. Policy sharpness: low entropy = decisive MCTS ──
        sharpness_s = 0.5  # Default if can't compute
        if game.policy_targets and game_len > 0:
            try:
                sample_idx = np.linspace(0, game_len - 1, min(5, game_len), dtype=int)
                entropies = []
                for i in sample_idx:
                    p = np.clip(game.policy_targets[i], 1e-8, 1.0)
                    entropies.append(-np.sum(p * np.log(p)))
                avg_entropy = np.mean(entropies)
                max_entropy = np.log(len(game.policy_targets[0]))  # log(441) ≈ 6.09
                sharpness_s = max(0.0, 1.0 - avg_entropy / max_entropy) if max_entropy > 0 else 0.5
            except Exception:
                pass
        score += self.W_SHARPNESS * sharpness_s

        # ── 5. Model maturity: games from later training steps are better ──
        step = meta.get('training_step', 0)
        maturity_s = min(1.0, step / 50000.0)
        score += self.W_MATURITY * maturity_s

        return score

    # ================================================================
    #  Game Insertion & Eviction
    # ================================================================

    @staticmethod
    def _precompute_focus_data(game: GameHistory):
        """Precompute sparse board snapshots and target_centers for every position.

        Called once at save_game() time.  Board snapshots are saved every
        SNAPSHOT_INTERVAL steps (like the existing board_snapshots dict)
        to avoid massive memory usage on long games (1400+ moves).
        Target centers are stored for ALL positions (negligible: 8 bytes each).
        """
        BOARD_SIZE = getattr(game, 'board_size', 100)
        VIEW_SIZE = 21
        HALF = VIEW_SIZE // 2
        SNAP_INTERVAL = GameHistory.SNAPSHOT_INTERVAL  # 10
        n = len(game)
        if n == 0:
            game.board_states = {}   # sparse dict: {step: int8(100,100)}
            game.target_centers_precomputed = []
            return

        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board_snapshots: Dict[int, np.ndarray] = {}
        target_centers = []

        for i in range(n):
            # Save sparse snapshot at interval boundaries
            if i % SNAP_INTERVAL == 0:
                board_snapshots[i] = board.copy()

            # Compute target center (global board coords of the actual move)
            act = game.actions[i]
            ctr = game.centers[i]
            lr = act // VIEW_SIZE
            lc = act % VIEW_SIZE
            tr = ctr[0] - HALF + lr
            tc = ctr[1] - HALF + lc
            tr = max(0, min(BOARD_SIZE - 1, tr))
            tc = max(0, min(BOARD_SIZE - 1, tc))
            target_centers.append(tr * BOARD_SIZE + tc)

            # Apply move to board (for next position's state)
            br = ctr[0] - HALF + lr
            bc = ctr[1] - HALF + lc
            pid = game.player_ids[i]
            if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE:
                board[br, bc] = pid

        game.board_states = board_snapshots  # sparse: ~1/10 of positions
        game.target_centers_precomputed = target_centers

    def save_game(self, game: GameHistory, training_step: int = 0):
        """Add a completed game with quality scoring and memory-aware eviction.
        Thread-safe: acquires _data_lock to protect games/meta lists."""
        # Precompute focus data BEFORE locking (CPU work, no shared state)
        if game.board_states is None:
            self._precompute_focus_data(game)
        with self._data_lock:
            self._save_game_locked(game, training_step)

    def _save_game_locked(self, game: GameHistory, training_step: int = 0):
        mem = self._estimate_game_memory(game)
        meta = {
            'insert_idx': self.total_games,
            'training_step': training_step,
            'memory_bytes': mem,
            'quality': 0.0,
            'timestamp': time.time(),
        }
        meta['quality'] = self._compute_quality(game, meta)

        self.total_games += 1
        self.games.append(game)
        self.meta.append(meta)
        self.total_memory += mem
        self._weights_dirty = True

        # Evict if over budget
        self._maybe_evict()

    def _maybe_evict(self):
        """Check if eviction is needed and run it."""
        needs_eviction = (
            self.total_memory > self.max_memory_bytes or
            len(self.games) > self.max_size
        )
        if needs_eviction:
            target_mem = int(self.max_memory_bytes * self.EVICTION_TARGET_RATIO)
            target_cnt = int(self.max_size * 0.90)
            reason = "memory" if self.total_memory > self.max_memory_bytes else "count"
            self._run_eviction(target_mem, target_cnt, reason=reason)
        else:
            # Safety valve: check system memory
            self._check_system_memory()

    def _run_eviction(self, target_memory: int, target_count: int, reason: str = "memory"):
        """Batch-evict lowest-quality games to reach target levels.

        Strategy:
        - Most recent 20% of games are protected (fresh data from latest model)
        - Among the remaining 80%, re-score quality and evict worst first
        - Never evict below min_games
        """
        n = len(self.games)
        if n <= self.min_games:
            return

        # Number of protected recent games
        n_protected = max(self.min_games, int(n * self.RECENT_PROTECT_RATIO))
        n_candidates = n - n_protected
        if n_candidates <= 0:
            return

        # Re-score candidates (older games) with updated recency
        scored = []
        for i in range(n_candidates):
            self.meta[i]['quality'] = self._compute_quality(self.games[i], self.meta[i])
            scored.append((i, self.meta[i]['quality'], self.meta[i]['memory_bytes']))

        # Sort by quality ascending (worst first)
        scored.sort(key=lambda x: x[1])

        # Collect indices to evict
        evict_set = set()
        freed = 0
        for idx, quality, mem in scored:
            remaining_n = n - len(evict_set)
            remaining_mem = self.total_memory - freed

            if remaining_mem <= target_memory and remaining_n <= target_count:
                break
            if remaining_n <= self.min_games:
                break

            evict_set.add(idx)
            freed += mem

        if not evict_set:
            return

        # Rebuild lists (O(n) but amortized infrequent)
        new_games = []
        new_meta = []
        for i in range(n):
            if i not in evict_set:
                new_games.append(self.games[i])
                new_meta.append(self.meta[i])

        evicted = len(evict_set)
        self._eviction_count += evicted
        self.games = new_games
        self.meta = new_meta
        self.total_memory -= freed
        self._weights_dirty = True

        # Report (reason: memory/count/emergency for interpretability)
        avg_quality_evicted = np.mean([scored[j][1] for j in range(evicted)]) if evicted > 0 else 0
        avg_quality_kept = np.mean([m['quality'] for m in self.meta]) if self.meta else 0
        print(f"[ReplayBuffer] Evicted {evicted} games (reason: {reason}, "
              f"avg_q={avg_quality_evicted:.3f}, freed {freed / (1024**2):.0f} MB). "
              f"Kept {len(self.games)} games "
              f"(avg_q={avg_quality_kept:.3f}, {self.total_memory / (1024**3):.2f} GB)",
              flush=True)

    def _check_system_memory(self):
        """Emergency eviction if system RAM is critically low."""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_gb < self.SYSTEM_RAM_EMERGENCY_GB and len(self.games) > self.min_games:
                _log.warning("EMERGENCY: System RAM low (%.1f GB). Emergency eviction...", available_gb)
                target_mem = int(self.total_memory * 0.70)
                target_cnt = int(len(self.games) * 0.70)
                self._run_eviction(target_mem, target_cnt, reason="emergency")
        except ImportError:
            pass  # psutil not available, rely on estimation-based limits

    # ================================================================
    #  Sampling
    # ================================================================

    def _get_sampling_weights(self) -> np.ndarray:
        """Compute normalized sampling weights for prioritized replay."""
        n = len(self.games)
        if n == 0:
            return np.array([], dtype=np.float64)

        if not self._weights_dirty and self._sampling_weights is not None:
            if len(self._sampling_weights) == n:
                return self._sampling_weights

        # Weight = quality^alpha * sqrt(game_length)
        # sqrt(len) gives longer games more representation without overwhelming
        weights = np.empty(n, dtype=np.float64)
        for i in range(n):
            q = max(0.01, self.meta[i].get('quality', 0.01))
            gl = max(1, len(self.games[i]))
            weights[i] = (q ** self.priority_alpha) * np.sqrt(gl)

        # Stability: avoid division by zero or numerical blowup when normalizing
        total = weights.sum()
        if total > 0:
            total = max(total, 1e-9)
            weights /= total
        else:
            weights[:] = 1.0 / n

        self._sampling_weights = weights
        self._weights_dirty = False
        return weights

    def sample_batch(self, batch_size: int, num_unroll_steps: int,
                     td_steps: int, discount: float,
                     action_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of positions with prioritized experience replay.
        Thread-safe: snapshots game list under lock, then processes without lock.

        Returns dict with:
            observations: (B, C, H, W)
            next_observations: (B, C, H, W)
            actions: (B, K) where K = num_unroll_steps
            target_values: (B, K+1)
            target_rewards: (B, K)
            target_policies: (B, K+1, action_size)
            global_states: (B, 4, 100, 100)
            target_centers: (B,)
        """
        # Snapshot game references under lock (fast — just copies list pointers)
        with self._data_lock:
            n = len(self.games)
            if n == 0:
                raise ValueError(
                "Cannot sample from empty replay buffer (num_games=0). "
                "Ensure buffer has games before calling sample_batch."
            )
            weights = self._get_sampling_weights()
            games_snapshot = list(self.games)  # shallow copy of references

        # Build batch: sample game indices and positions, then fill observations/targets/global_states per position.
        batch_obs = []
        batch_next_obs = []
        batch_actions = []
        batch_target_values = []
        batch_target_rewards = []
        batch_target_policies = []
        batch_global_states = []
        batch_target_centers = []
        batch_session_contexts = []
        batch_threats = []
        batch_heatmaps = []
        batch_opponent_actions = []
        batch_player_ids = []

        # Prioritized game selection (uses snapshot, no lock needed)
        game_indices = np.random.choice(n, size=batch_size, p=weights, replace=True)

        for gi in game_indices:
            game = games_snapshot[gi]
            if len(game) == 0:
                continue  # skip empty games (shouldn't happen, but defensive)
            pos = np.random.randint(len(game))

            # Observation at position
            batch_obs.append(game.observations[pos])

            # Next observation (t+1) for consistency loss
            if pos + 1 < len(game):
                batch_next_obs.append(game.observations[pos + 1])
            else:
                batch_next_obs.append(game.observations[pos])

            # Unroll targets
            actions = []
            target_values = []
            target_rewards = []
            target_policies = []

            # Value target for initial position
            target_values.append(
                self._compute_value_target(game, pos, td_steps, discount)
            )
            target_policies.append(game.policy_targets[pos])

            for step in range(num_unroll_steps):
                idx = pos + step
                if idx < len(game) - 1:
                    actions.append(game.actions[idx])
                    target_rewards.append(game.rewards[idx])
                    target_values.append(
                        self._compute_value_target(game, idx + 1, td_steps, discount)
                    )
                    target_policies.append(
                        game.policy_targets[idx + 1] if idx + 1 < len(game)
                        else np.zeros(action_size, dtype=np.float32)
                    )
                else:
                    actions.append(0)
                    target_rewards.append(0.0)
                    target_values.append(np.zeros(3, dtype=np.float32))
                    target_policies.append(np.zeros(action_size, dtype=np.float32))

            batch_actions.append(actions)
            batch_target_values.append(target_values)
            batch_target_rewards.append(target_rewards)
            batch_target_policies.append(target_policies)

            # --- Focus Network Data (sparse snapshot + short replay) ---
            BOARD_SIZE = getattr(game, 'board_size', 100)
            VIEW_SIZE = 21
            HALF = VIEW_SIZE // 2
            
            # Find nearest precomputed snapshot and replay at most ~SNAP_INTERVAL steps
            snapshots = game.board_states  # sparse dict {step: board}
            snap_interval = GameHistory.SNAPSHOT_INTERVAL
            snap_step = (pos // snap_interval) * snap_interval

            if isinstance(snapshots, dict) and snap_step in snapshots:
                current_board = snapshots[snap_step].copy()
                replay_start = snap_step
            elif isinstance(snapshots, list) and pos < len(snapshots):
                # Legacy per-position list format (backward compat)
                current_board = snapshots[pos]
                replay_start = pos  # no replay needed
            else:
                current_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
                replay_start = 0

            # Efficiency: replay from snapshot to pos (max ~SNAPSHOT_INTERVAL steps) instead of full game.
            for i in range(replay_start, pos):
                act = game.actions[i]
                ctr = game.centers[i]
                lr = act // VIEW_SIZE
                lc = act % VIEW_SIZE
                br = ctr[0] - HALF + lr
                bc = ctr[1] - HALF + lc
                pid = game.player_ids[i]
                if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE:
                    current_board[br, bc] = pid

            # Efficiency: vectorized global state (no per-cell Python loop).
            current_pid = game.player_ids[pos]
            next_pid = (current_pid % 3) + 1
            prev_pid = ((current_pid + 1) % 3) + 1

            glob_state = np.stack([
                (current_board == current_pid),
                (current_board == next_pid),
                (current_board == prev_pid),
                (current_board == 0),
            ]).astype(np.float32)
            batch_global_states.append(glob_state)

            # Target center (precomputed for every position)
            batch_target_centers.append(game.target_centers_precomputed[pos])

            # --- Session Context (Phase 4) ---
            # Compute per-position 4-dim context vector from game-level session data
            if game.session_scores is not None:
                pid = game.player_ids[pos]
                max_possible = max(1, game.session_length * 5)
                pids_sorted = sorted(game.session_scores.keys())
                others = [p for p in pids_sorted if p != pid]
                my_score = game.session_scores.get(pid, 0) / max_possible
                opp1_score = game.session_scores.get(others[0], 0) / max_possible if len(others) > 0 else 0.0
                opp2_score = game.session_scores.get(others[1], 0) / max_possible if len(others) > 1 else 0.0
                games_remaining = (game.session_length - game.session_game_idx - 1) / max(1, game.session_length)
                ctx = np.array([my_score, opp1_score, opp2_score, games_remaining], dtype=np.float32)
            else:
                ctx = np.zeros(4, dtype=np.float32)
            batch_session_contexts.append(ctx)
            
            # --- Auxiliary Targets (Phase 2) ---
            # 1. Threats
            # Pull threat vector from current position. Use zeros if missing (legacy)
            if hasattr(game, 'threats') and pos < len(game.threats):
                batch_threats.append(game.threats[pos])
            else:
                batch_threats.append(np.zeros(3, dtype=np.float32))

            # 2. Opponent Action (Future Action)
            # Predict action at pos+1. If terminal, use -1 (masked later) or 0? 
            # We'll valid mask it generally, but here let's store index.
            if pos + 1 < len(game):
                batch_opponent_actions.append(game.actions[pos+1])
            else:
                batch_opponent_actions.append(-1) # Terminal

            # 3. Board Heatmap (Next N=20 steps)
            heatmap = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.float32)
            lookahead = 20
            # iterate from pos+1 to pos+lookahead
            start_idx = pos + 1
            end_idx = min(len(game), start_idx + lookahead)
            
            ctr = game.centers[pos] # Heatmap is relative to CURRENT view center
            h_half = VIEW_SIZE // 2
            
            for k in range(start_idx, end_idx):
                ac = game.actions[k]
                c_ctr = game.centers[k] # absolute center of that move? NO.
                # Actions are stored as local indices 0-440 relative to THAT step's center.
                # We need to convert to absolute, then to local relative to CURRENT center.
                
                # Step 1: Recover absolute coordinate of future move
                lr = ac // VIEW_SIZE
                lc = ac % VIEW_SIZE
                
                # Center of step k
                if k < len(game.centers):
                    # Robustness check
                    k_ctr = game.centers[k]
                else:
                    k_ctr = ctr # should not happen given loop bounds
                    
                abs_r = k_ctr[0] - h_half + lr
                abs_c = k_ctr[1] - h_half + lc
                
                # Step 2: Convert to local coordinate relative to CURRENT center (pos)
                curr_min_r = ctr[0] - h_half
                curr_min_c = ctr[1] - h_half
                
                rel_r = abs_r - curr_min_r
                rel_c = abs_c - curr_min_c
                
                if 0 <= rel_r < VIEW_SIZE and 0 <= rel_c < VIEW_SIZE:
                    heatmap[rel_r, rel_c] = 1.0
            
            batch_heatmaps.append(heatmap)
            
            # KOTH Support: Add player_id of current move
            batch_player_ids.append(game.player_ids[pos])

        return {
            'observations': np.array(batch_obs, dtype=np.float32),
            'next_observations': np.array(batch_next_obs, dtype=np.float32),
            'actions': np.array(batch_actions, dtype=np.int64),
            'target_values': np.array(batch_target_values, dtype=np.float32),
            'target_rewards': np.array(batch_target_rewards, dtype=np.float32),
            'target_policies': np.array(batch_target_policies, dtype=np.float32),
            'global_states': np.array(batch_global_states, dtype=np.float32),
            'target_centers': np.array(batch_target_centers, dtype=np.int64),
            'session_contexts': np.array(batch_session_contexts, dtype=np.float32),
            'target_threats': np.array(batch_threats, dtype=np.float32),
            'target_opponent_actions': np.array(batch_opponent_actions, dtype=np.int64),
            'target_heatmaps': np.array(batch_heatmaps, dtype=np.float32),
            'player_ids': np.array(batch_player_ids, dtype=np.int64),
        }

    def _compute_value_target(self, game: GameHistory, pos: int,
                              td_steps: int, discount: float) -> np.ndarray:
        """Compute n-step TD value target (vector) for all players.
        n-step TD: V = sum_i gamma^i r_i + gamma^n V_bootstrap; rotate by player perspective.
        Returns:
            (3,) np.ndarray: [V_me, V_next, V_prev] from perspective of player at `pos`.
        """
        if pos >= len(game):
            # Terminal state
            return np.zeros(3, dtype=np.float32)

        my_player = game.player_ids[pos]
        # Map player ID to index 0,1,2 relative to me: [my, next, prev]
        # Helper to get next/prev player ID
        def next_p(pid): return (pid % 3) + 1
        def prev_p(pid): return ((pid - 2) % 3) + 1
        
        p_ids = [my_player, next_p(my_player), prev_p(my_player)]
        
        value = np.zeros(3, dtype=np.float32)

        for i in range(td_steps):
            idx = pos + i
            if idx < len(game):
                # Intermediate rewards are 0. Only terminal has placement rewards.
                if idx == len(game) - 1 and game.placement_rewards:
                    # Terminal step
                    rews = np.array([game.placement_rewards.get(pid, 0.0) for pid in p_ids], dtype=np.float32)
                    value += (discount ** i) * rews
                    return value # Terminal
            else:
                break

        bootstrap_idx = pos + td_steps
        if bootstrap_idx < len(game):
            # Bootstrap from N-step future root value.
            # root_values[bootstrap_idx] is [V_boot, V_boot+next, V_boot+prev].
            # We want [V_my, V_next, V_prev].
            # Shift = (bootstrap_player_idx - my_player_idx) % 3?
            # bootstrap_idx player is `game.player_ids[bootstrap_idx]`.
            # Let `boot_p` be player at bootstrap_idx.
            # `shift` is how many turns forward `boot_p` is from `my_player`.
            # If boot_p is next(my_player), shift=1. Val[0] is Next. We want it at index 1.
            # Val[2] is Me. We want it at index 0.
            # np.roll(Val, 1) -> [Val[2], Val[0], Val[1]]. Correct.
            # Calculate shift:
            # pid=1 -> 0, pid=2 -> 1, pid=3 -> 2.
            # shift = ( (boot_p - 1) - (my_player - 1) ) % 3
            boot_p = game.player_ids[bootstrap_idx]
            shift = (boot_p - my_player) % 3
            
            bootstrap_val = game.root_values[bootstrap_idx]
            rotated_val = np.roll(bootstrap_val, shift)
            
            value += (discount ** td_steps) * rotated_val

        return value

    # ================================================================
    #  Accessors
    # ================================================================

    def __len__(self):
        return len(self.games)

    def num_games(self) -> int:
        return len(self.games)

    def num_positions(self) -> int:
        return sum(len(g) for g in self.games)

    def memory_report(self) -> Dict[str, Any]:
        """Return memory usage statistics for logging/monitoring."""
        # Avoid O(n) over all games at scale; skip num_positions when above threshold (see SCALABILITY.md)
        n_pos = self.num_positions() if len(self.games) < self._REPORT_NUM_POSITIONS_MAX_GAMES else -1
        return {
            'num_games': len(self.games),
            'num_positions': n_pos,
            'total_memory_gb': self.total_memory / (1024 ** 3),
            'max_memory_gb': self.max_memory_bytes / (1024 ** 3),
            'usage_pct': 100.0 * self.total_memory / self.max_memory_bytes if self.max_memory_bytes > 0 else 0,
            'total_evictions': self._eviction_count,
            'avg_quality': float(np.mean([m['quality'] for m in self.meta])) if self.meta else 0.0,
            'avg_game_len': float(np.mean([len(g) for g in self.games])) if self.games else 0.0,
        }

    # ================================================================
    #  Persistence (backward-compatible)
    # ================================================================

    _SAVE_VERSION = 2

    def save(self, path: str):
        """Save buffer to disk (atomic: write to .tmp then replace)."""
        tmp_path = path + '.tmp'
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            data = {
                'version': self._SAVE_VERSION,
                'games': self.games,
                'meta': self.meta,
                'total_games': self.total_games,
                'total_memory': self.total_memory,
                'max_size': self.max_size,
                '_eviction_count': self._eviction_count,
            }
            with open(tmp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            _log.error("Failed to save: %s", e)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def load(self, path: str):
        """Load buffer from disk. Backward-compatible with old deque format."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            _log.error("Failed to load: %s", e)
            return

        version = data.get('version', 1)

        if version >= 2:
            self._load_v2(data)
        else:
            self._load_v1_compat(data)

        # Post-load: precompute focus data for old games, filter, enforce budget
        self._ensure_precomputed()
        self._filter_corrupted_games()
        self._recompute_memory()
        self._enforce_memory_budget()

        report = self.memory_report()
        print(f"[ReplayBuffer] Loaded {report['num_games']} games "
              f"({report['total_memory_gb']:.2f} GB / {report['max_memory_gb']:.1f} GB, "
              f"{report['usage_pct']:.1f}%, avg_q={report['avg_quality']:.3f})",
              flush=True)

    def _load_v2(self, data: Dict):
        """Load new format (version 2+). Only append entries that look like valid GameHistory."""
        raw_games = data.get('games', []) if isinstance(data.get('games'), list) else []
        raw_meta = data.get('meta', []) if isinstance(data.get('meta'), list) else []
        skipped = 0
        self.games = []
        self.meta = []

        for i, g in enumerate(raw_games):
            try:
                if not self._is_valid_game_entry(g):
                    skipped += 1
                    continue
                self.games.append(g)
                if i < len(raw_meta) and isinstance(raw_meta[i], dict):
                    self.meta.append(raw_meta[i])
                else:
                    self.meta.append({
                        'insert_idx': len(self.games) - 1,
                        'training_step': 0,
                        'memory_bytes': self._estimate_game_memory(g),
                        'quality': 0.5,
                        'timestamp': time.time(),
                    })
            except Exception:
                skipped += 1

        if skipped > 0:
            print(f"[ReplayBuffer] Skipped {skipped} invalid or corrupted game entries.", flush=True)

        self.total_games = data.get('total_games', len(self.games))
        self.max_size = data.get('max_size', self.max_size)
        self._eviction_count = data.get('_eviction_count', 0)
        self._weights_dirty = True

        # Ensure meta length matches games
        while len(self.meta) < len(self.games):
            idx = len(self.meta)
            self.meta.append({
                'insert_idx': self.total_games - len(self.games) + idx,
                'training_step': 0,
                'memory_bytes': self._estimate_game_memory(self.games[idx]) if idx < len(self.games) else 0,
                'quality': 0.5,
                'timestamp': time.time(),
            })

    def _load_v1_compat(self, data: Dict):
        """Load old format (deque-based, version 1)."""
        old_buffer = data.get('buffer', [])
        self.total_games = data.get('total_games', len(old_buffer))
        self.max_size = data.get('max_size', self.max_size)
        self._eviction_count = 0
        self._weights_dirty = True

        print(f"[ReplayBuffer] Converting v1 format ({len(old_buffer)} games)...", flush=True)

        self.games = []
        self.meta = []
        for i, game in enumerate(old_buffer):
            mem = self._estimate_game_memory(game)
            meta = {
                'insert_idx': i,
                'training_step': 0,
                'memory_bytes': mem,
                'quality': 0.5,  # Will be recomputed
                'timestamp': time.time(),
            }
            self.games.append(game)
            self.meta.append(meta)

        # Compute quality scores for all loaded games
        for i in range(len(self.games)):
            # Upgrade legacy scalar root_values to vector
            if len(self.games[i].root_values) > 0:
                first_val = self.games[i].root_values[0]
                if isinstance(first_val, (float, int, np.floating, np.integer)):
                    # Convert all scalars to [v, 0, 0]
                    # Note: scalar v was "current player win rate".
                    # [v, (1-v)/2, (1-v)/2] might be better but 0 is safe for now as 
                    # we rely on placement rewards for terminal anyway.
                    new_vals = []
                    for v in self.games[i].root_values:
                        vec = np.zeros(3, dtype=np.float32)
                        vec[0] = float(v)
                        new_vals.append(vec)
                    self.games[i].root_values = new_vals
                    
            self.meta[i]['quality'] = self._compute_quality(self.games[i], self.meta[i])

    def _is_valid_game_entry(self, g: Any) -> bool:
        """Return True if g looks like a valid GameHistory (has required list attrs)."""
        if g is None:
            return False
        for attr in ('observations', 'actions', 'rewards', 'policy_targets', 'root_values'):
            val = getattr(g, attr, None)
            if not isinstance(val, list):
                return False
        return True

    def _ensure_precomputed(self):
        """Backward compat: precompute focus data for old games missing it. Skip bad entries per-game."""
        count = 0
        skip_count = 0
        for game in self.games:
            try:
                if not self._is_valid_game_entry(game):
                    skip_count += 1
                    continue
                if getattr(game, 'board_states', None) is None:
                    self._precompute_focus_data(game)
                    count += 1
            except Exception:
                skip_count += 1
        if skip_count > 0:
            print(f"[ReplayBuffer] Skipped precompute for {skip_count} invalid game(s).", flush=True)
        if count > 0:
            print(f"[ReplayBuffer] Precomputed focus data for {count} legacy games.", flush=True)

    def _filter_corrupted_games(self):
        """Remove games with NaN/Inf data. Per-game try/except so one bad game doesn't kill the run."""
        valid_games = []
        valid_meta = []
        corrupted = 0
        meta_list = list(self.meta)
        while len(meta_list) < len(self.games):
            meta_list.append({'insert_idx': len(meta_list), 'training_step': 0, 'memory_bytes': 0, 'quality': 0.5, 'timestamp': time.time()})

        for game, meta in zip(self.games, meta_list):
            is_valid = True
            try:
                # Check observations
                obs_list = getattr(game, 'observations', None)
                if obs_list is None or not isinstance(obs_list, list):
                    is_valid = False
                else:
                    for obs in obs_list:
                        if np.isnan(obs).any() or np.isinf(obs).any():
                            is_valid = False
                            break

                # Check rewards
                if is_valid:
                    rewards_list = getattr(game, 'rewards', None)
                    if rewards_list is None or not isinstance(rewards_list, list):
                        is_valid = False
                    else:
                        for reward in rewards_list:
                            if np.isnan(reward) or np.isinf(reward):
                                is_valid = False
                                break

                # Check root values
                if is_valid:
                    rv_list = getattr(game, 'root_values', None)
                    if rv_list is None or not isinstance(rv_list, list):
                        is_valid = False
                    else:
                        for val in rv_list:
                            v = np.asarray(val)
                            if np.isnan(v).any() or np.isinf(v).any():
                                is_valid = False
                                break

                # Check policy targets
                if is_valid:
                    pt_list = getattr(game, 'policy_targets', None)
                    if pt_list is None or not isinstance(pt_list, list):
                        is_valid = False
                    else:
                        for p in pt_list:
                            if np.isnan(p).any() or np.isinf(p).any():
                                is_valid = False
                                break
            except Exception:
                is_valid = False

            if is_valid:
                valid_games.append(game)
                valid_meta.append(meta)
            else:
                corrupted += 1

        if corrupted > 0:
            _log.warning("Discarded %d corrupted games (NaN/Inf or invalid).", corrupted)

        self.games = valid_games
        self.meta = valid_meta
        self._weights_dirty = True

    def _recompute_memory(self):
        """Recompute total_memory from game-level estimates."""
        self.total_memory = 0
        for i, game in enumerate(self.games):
            mem = self._estimate_game_memory(game)
            self.meta[i]['memory_bytes'] = mem
            self.total_memory += mem

    def _enforce_memory_budget(self):
        """After loading, evict if over memory/count budget."""
        if self.total_memory > self.max_memory_bytes or len(self.games) > self.max_size:
            over_mem = self.total_memory - self.max_memory_bytes
            over_cnt = len(self.games) - self.max_size
            _log.warning(
                "Loaded buffer exceeds budget (mem: +%.0f MB, count: +%d). Running eviction...",
                max(0, over_mem) / (1024**2), max(0, over_cnt)
            )
            target_mem = int(self.max_memory_bytes * self.EVICTION_TARGET_RATIO)
            target_cnt = int(self.max_size * 0.90)
            reason = "memory" if self.total_memory > self.max_memory_bytes else "count"
            self._run_eviction(target_mem, target_cnt, reason=reason)
