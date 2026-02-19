# Reusability: public APIs and dependencies

How to reuse each component in other contexts (standalone scripts, custom training loops, server, tests). No dependency on train/train_async unless noted.

---

## Config contract (minimal attributes)

| Component | Entry point(s) | Config required? | Minimal attributes |
|-----------|----------------|------------------|---------------------|
| **game_env** | `EightInARowEnv(board_size, win_length)`, `reset()`, `step(row, col)` | No | Explicit args only. |
| **replay_buffer** | `ReplayBuffer(max_size, max_memory_gb, ...)`, `save_game`, `sample_batch(...)`, `save`/`load` | No | Explicit args only. |
| **mcts** | `gumbel_muzero_search(network, observation, legal_actions_mask, config, ...)` | Yes (config-like) | policy_size, num_simulations, discount, gumbel_*, mcts_batch_size (or getattr defaults). |
| **self_play** | `play_game(network, config, ...)`, `play_session(...)` | Yes (config-like) | board_size, win_length, num_players, local_view_size, temperature_drop_step, num_simulations, max_game_steps; optional: num_simulations_early/mid/late, policy_target_temp_*. |
| **data_augment** | `apply_board_augment(batch, rng, noise_std)` | No | Batch dict with observations, global_states, actions, etc. (see function). |
| **path_utils** | `safe_under(base_dir, path)`, `resolve_under(base_dir, path)` | No | None. |
| **log_utils** | `get_logger(name)` | No | None. |
| **muzero_config** | `MuZeroConfig`, `validate()` | — | Canonical config; for minimal reuse, use a small object with the attributes required by the component you call. |
| **curriculum** | `CurriculumManager(config)`, advance/record/check_graduation | Yes | config with curriculum_* and checkpoint_dir. |
| **league** | `LeagueManager(config)`, get_opponent, add_opponent, record_match | Yes | config.checkpoint_dir; league file path. |
| **pbt** | `Population(config)`, load/save, evolve | Yes | config (pbt_*, checkpoint_dir). |
| **muzero_network** | `MuZeroNetwork.from_config(config)`, representation, dynamics, prediction | Yes | config for architecture (observation_channels, hidden_state_dim, policy_size, etc.). |
| **server** | FastAPI `app`, `load_model(path)` | Yes | config for validate and network; model path. |
| **train / train_async** | Main entry scripts | Yes | Full MuZeroConfig. Orchestrators; reusers can compose a subset (e.g. env + MCTS + replay only) for a custom loop. |

---

## Per-component summary

- **game_env**: `EightInARowEnv(board_size, win_length)`, `reset()`, `step(row, col)`. Dependencies: numpy only. Reusable in: any script (tests, custom loops, server for legal-move checks).

- **replay_buffer**: `ReplayBuffer(max_size, max_memory_gb, priority_alpha, min_games)`, `save_game(game)`, `sample_batch(batch_size, num_unroll_steps, td_steps, discount, action_size)`, `save`/`load`. Dependencies: none on MuZeroConfig; constructor and sample_batch take explicit parameters. Reusable in: any training loop that needs prioritized experience replay with memory budget. `GameHistory` is a plain data container.

- **mcts**: `gumbel_muzero_search(network, observation, legal_actions_mask, config, ...)`. Dependencies: config-like object with at least policy_size, num_simulations, discount, gumbel_*, mcts_batch_size (or getattr defaults). Network must provide initial_inference, recurrent_inference. Reusable in: self-play, server, or any script that needs MCTS policy and value.

- **self_play**: `play_game(network, config, ...)`, `play_session(...)`. Dependencies: config-like object with board_size, win_length, num_players, local_view_size, temperature_drop_step, num_simulations (and optional num_simulations_early/mid/late), policy_target_temp_*, max_game_steps; network with predict_center, initial_inference, recurrent_inference. Reusable in: train, train_async, or a custom loop that produces GameHistory trajectories.

- **data_augment**: `apply_board_augment(batch, rng, noise_std)`. Dependencies: batch dict with observations, global_states, actions, target_policies, etc. (see function). Reusable in: any pipeline with the same batch shape contract.

- **path_utils**: `safe_under(base_dir, path)`, `resolve_under(base_dir, path)`. Dependencies: none. Reusable everywhere.

- **log_utils**: `get_logger(name)`. Dependencies: none. Reusable everywhere.

- **muzero_config**: `MuZeroConfig` dataclass and `validate()`. Reusable as the canonical config; for minimal reuse (e.g. MCTS-only), a small object or dict with the needed attributes is enough (see mcts/self_play rows).

- **curriculum**: `CurriculumManager(config)`, advance, record_game_result, record_loss, check_graduation. Requires config and optional league. Reusable in train or train_async-style loops.

- **league**: `LeagueManager(config)`, get_opponent, add_opponent, record_match. Requires config.checkpoint_dir and league file. Reusable in train or train_async-style loops.

- **pbt**: `Population(config)`, load, save, evolve. Requires config (pbt_*, checkpoint_dir). Reusable in train_async-style loops.

- **muzero_network / transformer_backbone / consistency / engram**: `MuZeroNetwork.from_config(config)`; representation, dynamics, prediction, engram_module, consistency. Dependencies: config for architecture params. Reusable in any code that needs the same network architecture.

- **server**: FastAPI app and load_model; depends on config and model path. Reusable as a separate process or mountable app.

- **train / train_async**: Orchestrators that compose the above; main entry points. They use config, replay_buffer, self_play, mcts, curriculum, league, etc. Reusers can compose a subset (e.g. env + MCTS + replay only) for a custom loop without running train/train_async.
