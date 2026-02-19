# Maintainability: per-component notes

When editing any component, preserve the invariants and safeguards listed below. See also the robustness checklist in [train.py](train.py) docstring.

Scale-related knobs and limits: see [ai/SCALABILITY.md](SCALABILITY.md). Reusability: public APIs and dependencies — see [ai/REUSABILITY.md](REUSABILITY.md). Efficiency: existing measures and what to preserve — see [ai/EFFICIENCY.md](EFFICIENCY.md). Readability: standards and per-component notes — see [ai/READABILITY.md](READABILITY.md).

| Component | Role | Do not remove |
|-----------|------|----------------|
| **train** | Synchronous training loop: self-play, replay, train_step, checkpoint, curriculum, league. | Resume in try/except; train_step batch key/tensor validation, NaN/Inf checks, target clamp, loss/grad NaN guard, max_grad_norm; warmup when warmup_steps > 0; config.validate() before run. |
| **train_async** | Async training: CPU self-play actors, GPU learner, shared memory sync. | Full resume in try/except; config.validate() at entry; NaN streak detection and re-sanitize; queue.Full handling (log when repeated); checkpoint/save retry. |
| **server** | FastAPI server for AI move endpoint. | load_model in try/except; get_move validates board/player/cells; inference wrapped in try/except with random-legal fallback; path validation for checkpoint load. |
| **replay_buffer** | Experience replay with quality scoring and memory budget. | _load_v2 entry validation; _ensure_precomputed / _filter_corrupted_games for bad games; sampling weight sum uses max(sum, 1e-9); save/load retry. |
| **game_env** | Eight-in-a-row environment (step, reset, observations). | step try/except and retry; action/center validation; board_size/win_length validation in __init__. |
| **self_play** | Self-play game generation (play_game, play_session). | Step try/except and retry; value/reward sanitization; consecutive failure handling. |
| **mcts** | Gumbel MuZero MCTS (gumbel_muzero_search). | Value/reward sanitization; action validation. |
| **muzero_config** | Configuration dataclass and validation. | validate() (board_size, win_length, learning_rate, batch_size, checkpoint_dir under cwd); callers must call config.validate() at entry. |
| **path_utils** | Path safety for file I/O. | safe_under, resolve_under — used for checkpoint_dir and any user-provided paths. |
| **log_utils** | Structured logging (get_logger). | Use get_logger(__name__) in modules for errors/warnings. |
| **curriculum** | Per-stage board size/win length with graduation. | Graduation thresholds and elasticity (curriculum_max_games, base_games); stage_idx clamp; load/save with .get and skip bad entries. |
| **league** | Historical opponents with Elo; sampling and eviction. | Load/save retry; empty pool handling (get_opponent returns None); skip bad entries on load. |
| **pbt** | Population-based training (population of agents, evolutionary updates). | Load/save retry; skip bad entries on load. |
| **board_render** | Render board to image (e.g. PNG). | Save retry; handle missing matplotlib. |
| **data_augment** | Board augmentation (rotation, mirror, optional noise). | Entry validation for batch keys; action index bounds. |
| **muzero_network** | MuZero networks (representation, dynamics, prediction, engram, consistency). | — |
| **transformer_backbone** | DeepSeek-style MLA Transformer backbone. | — |
| **consistency** | EfficientZero consistency loss (SimSiam-style). | — |
| **engram** | External episodic memory (MemoryBank). | — |
