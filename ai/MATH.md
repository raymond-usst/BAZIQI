# Mathematical model (MuZero / Gumbel MuZero)

This document states the mathematical contracts implemented in the codebase: abstraction (state, action, value), accuracy of formulas (Bellman/Q, n-step TD, losses), and deduction (value propagation, player rotation).

---

## State and action

- **State**: Board of size S×S (e.g. 100×100), integer cells 0=empty, 1/2/3 = players; current player index 0/1/2 (players 1, 2, 3). Observation is a local view (e.g. 21×21) plus global thumbnail, from current player's perspective.
- **Action**: Local action index in [0, policy_size) with policy_size = local_view_size² (e.g. 21×21 = 441). Converted to board (row, col) via a fixed mapping from view center. **Action space** is a single source of truth: `policy_size = 21*21` in config; env, MCTS, replay buffer, and train all use the same local action index in `[0, policy_size)`.
- **Dynamics**: Environment step is deterministic: place piece, check win (first to WIN_LENGTH in a row), else next player.
- **Step reward**: Scalar for the current player only; 1.0 on win, 0.0 otherwise. Intermediate steps contribute 0 to the value target sum — only the terminal step uses `placement_rewards` for the value vector.

---

## Value semantics

- **Value vector**: (V_me, V_next, V_prev) = expected discounted return for each player from the current state, in "current player's perspective" (me = current, next/prev = opponents in turn order).
- **Range**: In practice from placement_rewards and bootstrap MCTS values; typically in [-1, 1] or [0, 1] depending on the head and clamping.
- **At terminal**: Value target uses placement_rewards (e.g. 1st: +1, 2nd: -0.2, 3rd: -1) per player for the vector.
- **Network output**: The value head outputs a 3-vector (V_me, V_next, V_prev) in the same order as the n-step TD target; training uses `target_values[:, 0]` as the vector target for the root step.

---

## MCTS (Gumbel MuZero)

- **Q-formula**: Q(s, a) = r + γ·V(s′) where r = `child.reward` (immediate reward from dynamics), V(s′) = `child.value()` (vector mean at child). Used for action scoring at root and interior.
- **Backup**: Monte Carlo style — the leaf value (or bootstrap value) is added to every node on the search path. The mean at a node is the average of leaf returns over all simulations passing through that node.
- **Policy improvement**: Improved policy = softmax(logits + c_scale · normalized_Q) (Gumbel MuZero); normalized_Q uses MinMaxStats over the tree.

---

## Training

- **Value target (n-step TD)**:  
  V_target(pos) = Σ_{i=0}^{min(td_steps, to_terminal)-1} γ^i · r_{pos+i} + γ^td_steps · V_bootstrap  
  when a bootstrap step exists; otherwise terminal sum. Rewards r_i are per-player at terminal (placement_rewards); bootstrap value is rotated by player perspective via `np.roll(bootstrap_val, shift)` so that [V_me, V_next, V_prev] matches the player at `pos`. Computed by `_compute_value_target`; stored as the vector target for value loss.
- **Reward target**: Scalar `game.rewards[idx]` (step reward) for each position; used for reward loss only. Value target is the vector from `_compute_value_target` (n-step TD with terminal placement_rewards or bootstrap).
- **Losses**:
  - Value / reward: MSE to target (vector value from n-step TD; scalar reward from step reward).
  - Policy: Cross-entropy to MCTS improved policy, i.e. -Σ target · log_softmax(logits).
  - Consistency (EfficientZero): 2 - 2·cos_sim between predicted and actual projected hidden states (SimSiam-style).
- **Gradient scale**: Unroll gradient scaled by 1/K (K = num_unroll_steps) per MuZero paper.

---

## Game

- **Win**: First player to get WIN_LENGTH in a row (horizontal, vertical, or diagonal). `step()` returns (reward, done); reward = 1.0 for win, 0.0 otherwise (scalar for current player).
- **Placement rewards**: At terminal, rank_players() sets placement_rewards (1st/2nd/3rd) used for value targets (vector per player).

---

## Contract summary

- **Env**: Step returns (reward, done); reward is scalar for current player; action is local index in [0, policy_size).
- **MCTS**: Q(s,a) = r + γ·V(s′); backup adds leaf value to path; improved policy = softmax(logits + c_scale·normalized_Q).
- **Replay**: n-step TD value target (vector) via `_compute_value_target`; scalar reward target from `game.rewards[idx]`.
- **Train**: MSE to value and reward targets; CE to MCTS policy; consistency 2−2·cos_sim; unroll losses scaled by 1/K.
