# Scalability: knobs and limits

Scale-related configuration and trade-offs. Tune these when increasing actors, buffer size, batch size, or board size.

---

## Actors / throughput (train_async)

- **`--actors`**: Number of self-play actor processes. Default: `min(8, cpu_count // 4)`. Increasing actors increases game throughput.
- **`game_queue` maxsize**: Config field `game_queue_maxsize` (default 200). Games produced by actors are put here; the learner consumes. If the learner is slow, the queue can fill and actors block until space. Consider increasing when using many actors.
- **`live_queue` maxsize**: Config field `live_queue_maxsize` (default 2000). Live events for dashboard; larger if many actors and high event rate.

---

## Replay buffer

- **`replay_buffer_size`**: Maximum number of games (hard cap). Validated in config (e.g. 1–2_000_000).
- **`max_memory_gb`**: Memory budget in GB for buffer contents. Validated (e.g. 0.1–500). Eviction runs when over budget.
- **`min_buffer_size`** / **`min_buffer_games`**: Minimum games to keep; eviction will not go below this. Must be ≤ `replay_buffer_size`.

Larger buffer = more diversity but more RAM and potentially slower sampling when `num_games` is very large. `memory_report()` skips expensive `num_positions()` when game count is above a threshold to avoid O(n) cost at scale.

---

## Batch size

- **`batch_size`**: Training batch size (validated 1–10000). Larger = better GPU utilization but more VRAM and memory per step. Override with `--batch-size` in train_async.

---

## Board / game size

- **`board_size`**: Board dimension (validated 3–200). Larger boards = longer games and bigger observations.
- **`win_length`**: Must be in [2, board_size].
- **`max_game_steps`**: Config field (default 5000). Self-play caps each game at this many steps to avoid runaway games; configurable so you can raise it for very large boards if needed.

---

## Queues (train_async)

- **`game_queue_maxsize`**: Max number of completed games in the queue between actors and learner. Backpressure when full.
- **`live_queue_maxsize`**: Max live events for the first actor’s dashboard stream.

Both are in config so operators can tune without editing code.

---

## Prefetch (train_async learner)

- **`prefetch_workers`**: Number of worker threads for the batch prefetcher (default 2). Double-buffers to reduce GPU stall. Can be tuned for I/O-bound vs compute-bound workloads.
