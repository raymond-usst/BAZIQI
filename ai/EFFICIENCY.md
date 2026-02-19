# Efficiency: existing measures and what to preserve

What each component already does for performance. Preserve these when editing.

---

## train_async

- **Prefetch**: Next batch is sampled and converted to pinned tensors on a background thread (ThreadPoolExecutor) so GPU computation overlaps with data transfer.
- **pin_memory**: Batches are pinned before being sent to GPU for async DMA.
- **non_blocking**: train_step moves tensors to device with non_blocking=True when batch is prefetched (in train.py).
- **Tuning**: Use prefetch_workers and batch_size for GPU utilization; see [SCALABILITY.md](SCALABILITY.md).

---

## train

- **Sync loop**: No prefetch; each step does sample_batch then train_step. Use train_async for higher throughput.
- **train_step**: Accepts both numpy and pre-converted (pinned) tensors; uses non_blocking when moving to device so train_async can overlap transfer with compute.

---

## replay_buffer

- **Sparse snapshots**: Board state is stored every SNAPSHOT_INTERVAL steps (not every step), reducing memory.
- **Short replay gap**: When building global_states for a position, replay runs only from the nearest snapshot to pos (max ~SNAPSHOT_INTERVAL steps), not the full game (1400+ steps).
- **Vectorized global state**: 4-channel global state is built with np.stack and boolean comparisons (no per-cell Python loop).
- **Eviction and memory budget**: Enforced to stay within max_memory_gb; see SCALABILITY.md.
- **memory_report**: num_positions() is skipped when game count exceeds a threshold to avoid O(n) at scale; see SCALABILITY.md.

---

## mcts

- **Batched leaf evaluation**: _batch_simulate_phase evaluates multiple leaves in one recurrent_inference call (config.mcts_batch_size) to reduce device round-trips.
- **torch.no_grad()**: All inference (initial_inference, recurrent_inference) runs under no_grad.
- **Virtual loss**: Used for tree diversity within each batch.

---

## self_play

- **Optional**: Config attributes (num_simulations_early/mid/late, temperature_drop_step, policy_target_temp_*, max_game_steps) can be cached at game start and used in the step loop to avoid repeated getattr.

---

## game_env

- **Vectorized plane updates** and comparisons where applicable.
- **Chain counting**: O(4*Board²); documented as "very fast" in code.

---

## data_augment

- **In-place**: apply_board_augment modifies the batch dict in place; no extra copy.

---

## muzero_network / transformer_backbone

- Standard PyTorch forward; no redundant copies. eval() and no_grad in inference paths.

---

## engram

- Pre-allocated key/value tensors; no per-write allocation.

---

## server

- Single model load at startup. Inference under eval mode; use no_grad in get_move if not already.
