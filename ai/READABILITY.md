# Readability: standards and per-component notes

What we do for clarity and what to preserve when editing. See [ai/MAINTENANCE.md](MAINTENANCE.md) for invariants.

---

## Standards

- **Module docstring**: Purpose (and optional "See ai/MAINTENANCE.md"). One place to see what the file does.
- **Class docstring**: Role and main invariants so callers know how to use it.
- **Public function docstring**: One-line or Args/Returns so the API is clear.
- **Long functions** (e.g. >80 lines): Section comments (e.g. `# ── Section ──` or `# 1. Step`) so the flow is scannable without reading every line.
- **Named constants**: Use constants instead of magic numbers where they clarify intent (e.g. SNAPSHOT_INTERVAL, OPTIMAL_LEN_MIN).

---

## Per-component summary

| Component | Readability notes |
|-----------|-------------------|
| **train** | Module + robustness checklist; train_step sectioned (device, validation, forward, losses, backward); save_checkpoint / save_metrics_log / load_metrics_log documented. |
| **train_async** | Module docstring; actor_loop / learner_loop with overview comment; SharedStats documented; helper docstrings. |
| **server** | Module docstring; MoveRequest / MoveResponse / StatusResponse; load_model docstring; get_move validated and wrapped. |
| **replay_buffer** | Module + features list; GameHistory and ReplayBuffer class docstrings; store() documented; sample_batch sectioned; named constants. |
| **game_env** | Module docstring; EightInARowEnv class and method docstrings; section comments where needed. |
| **self_play** | Module docstring; play_game / play_session docstrings; config cache comment. |
| **mcts** | Module + algorithm summary; MinMaxStats / MCTSNode; gumbel_muzero_search Args; section headers (Data Structures, Gumbel MuZero Search). |
| **muzero_config** | Module docstring; MuZeroConfig class docstring; validate() docstring; field comments. |
| **path_utils** | Module docstring; safe_under / resolve_under docstrings. |
| **log_utils** | Module docstring; get_logger docstring. |
| **curriculum** | Module docstring; CurriculumStage / CurriculumManager docstrings. |
| **league** | Module docstring; LeagueOpponent / LeagueManager docstrings. |
| **pbt** | Module docstring; PBTAgent / Population docstrings. |
| **board_render** | Module docstring; board_to_image_path docstring; _wood_background documented. |
| **data_augment** | Module docstring; apply_* and _rot/_inv docstrings. |
| **muzero_network** | Module docstring; DynamicsNetwork / PredictionNetwork / FocusNetwork / MuZeroNetwork docstrings. |
| **transformer_backbone** | Module docstring; class docstrings (RotaryEmbedding, MLA, SwiGLUFFN, DeepNormTransformerBlock, PatchEmbedding, TransformerBackbone). |
| **consistency** | Module + reference; ConsistencyModule and forward docstring. |
| **engram** | Module docstring; MemoryBank / EngramModule docstrings. |
