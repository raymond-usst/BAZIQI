"""MuZero configuration for Eight-in-a-Row.

Maintainability: see ai/MAINTENANCE.md."""

import os
import dataclasses

from ai.path_utils import safe_under

@dataclasses.dataclass
class MuZeroConfig:
    """Dataclass for game, network, replay, MCTS, and training. Call validate() before training or server. See field comments and SCALABILITY.md for scale-related fields."""
    # --- Game (Curriculum Learning) ---
    board_size: int = 100       # [15, 30, 50, 100]
    win_length: int = 8         # [5, 6, 7, 8]
    num_players: int = 3
    local_view_size: int = 21   # odd number, centered on action region

    # --- Transformer Backbone (DeepSeek MLA) ---
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 12
    d_kv_compress: int = 64
    ffn_hidden: int = 512
    dropout: float = 0.1
    patch_size: int = 3

    # --- Network Standard ---
    observation_channels: int = 8    # 4 local + 4 global thumbnail
    hidden_state_dim: int = 128     # increased from 64
    policy_size: int = 21 * 21      # local_view_size^2
    fc_hidden: int = 512            # increased from 256

    # --- Engram Memory ---
    use_engram: bool = True
    memory_capacity: int = 10000
    memory_top_k: int = 8
    memory_heads: int = 4

    # --- EfficientZero Consistency ---
    use_consistency: bool = True
    lambda_consistency: float = 2.0
    consistency_proj_dim: int = 256

    # --- Gumbel MuZero MCTS ---
    num_simulations: int = 100      # default (mid-game) during training — 2x for cleaner policy targets
    num_simulations_play: int = 100 # during human vs AI
    num_simulations_early: int = 32  # opening (step < 10): fewer sims, simple positions
    num_simulations_mid: int = 100   # mid-game (10-40): full depth search
    num_simulations_late: int = 50   # late-game (40+): positions more determined
    
    # Gumbel parameters
    gumbel_max_considered_actions: int = 16
    gumbel_c_visit: int = 50
    gumbel_c_scale: float = 1.0
    mcts_batch_size: int = 8             # batch size for virtual-loss batched MCTS

    # Old MCTS params (kept for compatibility/fallback)
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    temperature: float = 1.0
    temperature_drop_step: int = 30

    # Policy target temperature annealing
    policy_target_temp_start: float = 1.5   # soft targets early (explore)
    policy_target_temp_end: float = 0.5     # sharp targets late (exploit)
    policy_target_temp_steps: int = 100000  # anneal over this many steps

    # Self-Play League
    league_save_interval: int = 5000        # save league snapshot every N steps
    league_max_snapshots: int = 10          # keep last N league snapshots
    league_opponent_prob: float = 0.2       # probability of using league opponent

    # Phase 6: King-of-the-Hill (KOTH)
    koth_mode: bool = False
    koth_period: int = 10000        # switch active trainer every N steps

    # --- Training ---
    batch_size: int = 512           # large batch for GPU saturation (RTX 4090 16GB)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_decay_steps: int = 100000
    lr_decay_rate: float = 0.1
    training_steps: int = 100000
    warmup_steps: int = 0           # LR linear warmup (0 = disabled); e.g. 1000 to match async
    max_grad_norm: float = 1.0      # gradient clipping max norm (tunable for stability)
    clip_value_reward: bool = True  # clamp value/reward outputs in network forward (stability)
    checkpoint_interval: int = 500
    selfplay_games_per_iter: int = 10 # reduced due to slower MCTS
    training_steps_per_iter: int = 100
    num_unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 1.0

    # --- Data Augmentation ---
    augment_board: bool = True   # rotation (0/90/180/270°) + horizontal mirror on sample
    augment_noise_std: float = 0.0  # if > 0, add Gaussian noise to observations (e.g. 0.02)

    # --- Replay Buffer ---
    replay_buffer_size: int = 50000
    min_buffer_size: int = 500
    max_memory_gb: float = 35.0         # Memory budget for replay buffer (leave ~10GB for model/system)
    priority_alpha: float = 0.6         # Prioritized sampling sharpness (0=uniform, 1=proportional)
    min_buffer_games: int = 100         # Minimum games to keep (won't evict below this)

    # --- Scale / train_async ---
    max_game_steps: int = 5000          # Cap per-game steps in self-play to avoid runaway (see SCALABILITY.md)
    game_queue_maxsize: int = 200       # Max games in queue between actors and learner (backpressure)
    live_queue_maxsize: int = 2000      # Max live events for dashboard
    prefetch_workers: int = 2           # Learner batch prefetcher thread count

    # --- Session (Multi-Game) ---
    session_length: int = 5              # games per session (adaptive during training)
    session_length_min: int = 3          # min session length (early training)
    session_length_max: int = 7          # max session length (late training)
    placement_rewards: tuple = (1.0, -0.2, -1.0)  # 1st/2nd/3rd reward mapping
    placement_points: tuple = (5, 2, 0)             # 1st/2nd/3rd point values

    # --- KOTH (King of the Hill) ---
    koth_mode: bool = False
    koth_period: int = 1000         # Steps before rotating active player

    # --- PBT (Population Based Training) ---
    pbt_population_size: int = 4
    pbt_period: int = 5000          # Steps between evolutionary events
    pbt_mutation_rate: float = 0.2  # Probability of mutating a hyperparam

    # --- Misc ---
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 4

    def validate(self) -> None:
        """Check that key parameters are in valid ranges. Raises ValueError with a clear message if not."""
        if self.board_size < 3 or self.board_size > 200:
            raise ValueError(
                f"board_size must be in [3, 200], got {self.board_size}. "
                "Use a positive integer within this range."
            )
        if self.win_length < 2 or self.win_length > self.board_size:
            raise ValueError(
                f"win_length must be in [2, board_size] (board_size={self.board_size}), got {self.win_length}."
            )
        if self.learning_rate <= 0 or not (1e-6 <= self.learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate should be in [1e-6, 1.0], got {self.learning_rate}."
            )
        if self.batch_size < 1 or self.batch_size > 10000:
            raise ValueError(
                f"batch_size should be in [1, 10000], got {self.batch_size}."
            )
        if self.replay_buffer_size < 1 or self.replay_buffer_size > 2_000_000:
            raise ValueError(
                f"replay_buffer_size should be in [1, 2_000_000], got {self.replay_buffer_size}."
            )
        if self.max_memory_gb < 0.1 or self.max_memory_gb > 500:
            raise ValueError(
                f"max_memory_gb should be in [0.1, 500], got {self.max_memory_gb}."
            )
        if self.min_buffer_size > self.replay_buffer_size:
            raise ValueError(
                f"min_buffer_size ({self.min_buffer_size}) must be <= replay_buffer_size ({self.replay_buffer_size})."
            )
        if self.min_buffer_games > self.replay_buffer_size:
            raise ValueError(
                f"min_buffer_games ({self.min_buffer_games}) must be <= replay_buffer_size ({self.replay_buffer_size})."
            )
        if not safe_under(os.getcwd(), self.checkpoint_dir):
            raise ValueError(
                f"checkpoint_dir must resolve to a path under the current directory, got {self.checkpoint_dir!r}."
            )
