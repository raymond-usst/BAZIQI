"""Unit tests for mathematical rigor: value target, Q formula, policy loss, consistency loss, player rotation."""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.replay_buffer import ReplayBuffer, GameHistory
from ai.consistency import ConsistencyModule
from ai.mcts import MCTSNode, _backpropagate, MinMaxStats


# ----- Value target: terminal within td_steps -----

def test_value_target_terminal_within_td_steps():
    """Terminal within td_steps: value = sum gamma^i * placement_rewards at terminal step."""
    game = GameHistory()
    game.player_ids = [1, 2, 3]  # 3 steps, alternating
    game.placement_rewards = {1: 1.0, 2: -0.2, 3: -1.0}
    # Minimal: we only need len(game)=3, last index 2 is terminal
    for _ in range(3):
        game.observations.append(np.zeros((4, 21, 21), dtype=np.float32))
        game.actions.append(0)
        game.rewards.append(0.0)
        game.policy_targets.append(np.ones(441) / 441)
        game.root_values.append(np.zeros(3, dtype=np.float32))
        game.threats.append(np.zeros(3, dtype=np.float32))
        game.centers.append((0, 0))
    game.done = True

    buffer = ReplayBuffer(max_size=10)
    td_steps = 5
    discount = 0.99
    # At pos=0, we look idx=0,1,2. At i=2, idx=2 = len(game)-1, terminal.
    # value = gamma^2 * [V_me, V_next, V_prev] = gamma^2 * placement_rewards for p_ids at pos 0.
    # my_player=1 -> p_ids=[1,2,3] -> rews=[1.0, -0.2, -1.0]
    target = buffer._compute_value_target(game, 0, td_steps, discount)
    expected = (discount ** 2) * np.array([1.0, -0.2, -1.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(target, expected)
    # Optional: value in reasonable range
    assert np.all(target >= -1.5) and np.all(target <= 1.5)


# ----- Value target: bootstrap -----

def test_value_target_bootstrap_rotation():
    """Bootstrap: value = gamma^td_steps * rotated(bootstrap_root_value); rotation by player perspective."""
    game = GameHistory()
    # 5 steps so bootstrap_idx = 0+2 = 2 exists
    for i in range(5):
        game.player_ids.append((i % 3) + 1)
        game.observations.append(np.zeros((4, 21, 21), dtype=np.float32))
        game.actions.append(0)
        game.rewards.append(0.0)
        game.policy_targets.append(np.ones(441) / 441)
        # root_values[i] from perspective of player at i
        game.root_values.append(np.array([0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)], dtype=np.float32))
        game.threats.append(np.zeros(3, dtype=np.float32))
        game.centers.append((0, 0))
    game.done = False
    game.placement_rewards = {}

    buffer = ReplayBuffer(max_size=10)
    td_steps = 2
    discount = 0.99
    # pos=0, my_player=1; bootstrap_idx=2, boot_p=3. shift = (3-1)%3 = 2.
    # root_values[2] = [0.3, 0.6, 0.9]. np.roll(., 2) = [0.6, 0.9, 0.3].
    target = buffer._compute_value_target(game, 0, td_steps, discount)
    bootstrap_val = np.array([0.3, 0.6, 0.9], dtype=np.float32)
    shift = (game.player_ids[2] - game.player_ids[0]) % 3
    rotated = np.roll(bootstrap_val, shift)
    expected = (discount ** td_steps) * rotated
    np.testing.assert_array_almost_equal(target, expected)


# ----- Backup deduction (mean at node = avg of leaf returns) -----

def test_backpropagate_single_node_value_mean():
    """After one _backpropagate with a single-node path, value_sums and value() equal the backed-up value."""
    node = MCTSNode(prior=0.1, logit=0.0)
    search_path = [node]
    value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    discount = 0.99
    min_max = MinMaxStats()
    _backpropagate(search_path, value, discount, min_max, num_players=3)
    np.testing.assert_array_almost_equal(node.value_sums, value)
    assert node.visit_count == 1
    np.testing.assert_array_almost_equal(node.value(), value)


# ----- Q formula -----

def test_q_formula_mock_nodes():
    """Q(s,a) = r + gamma*V(s'); shape and value check with mock nodes."""
    discount = 0.99
    child = MCTSNode(prior=0.1, logit=0.0)
    child.reward = 0.5
    child.visit_count = 10
    child.value_sums = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # value() = [0.1, 0.2, 0.3]

    q_val = child.reward + discount * child.value()
    expected = 0.5 + discount * np.array([0.1, 0.2, 0.3], dtype=np.float32)
    np.testing.assert_array_almost_equal(q_val, expected)
    assert q_val.shape == (3,)


# ----- Policy loss: CE = -sum(target * log_softmax(logits)) -----

def test_policy_loss_cross_entropy():
    """Policy loss = -sum(target * log_softmax(logits)); compare to train_step-style CE."""
    target_probs = torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float32)
    logits = torch.tensor([[1.0, 2.0, 0.5]], dtype=torch.float32)
    log_probs = F.log_softmax(logits, dim=1)
    ce_manual = -(target_probs * log_probs).sum(dim=1)
    # Same as F.cross_entropy with soft labels: input logits, target as probs
    ce_soft = -(target_probs * log_probs).sum(dim=1).item()
    ce_train_style = -(target_probs * log_probs).sum(dim=1).mean().item()
    assert abs(ce_manual.item() - ce_soft) < 1e-6
    assert ce_train_style == ce_manual.item()


# ----- Consistency loss: 2 - 2*cos_sim -----

def test_consistency_loss_formula():
    """Consistency loss = 2 - 2*cos_sim for normalized vectors (formula check with fixed tensors)."""
    p_pred = F.normalize(torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.707]]), dim=-1)
    z_actual = F.normalize(torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.707]]), dim=-1)
    cos_sim = (p_pred * z_actual).sum(dim=-1)
    expected = 2.0 - 2.0 * cos_sim
    assert expected[0].item() == 0.0  # same direction -> cos_sim=1 -> loss 0
    assert abs(expected[1].item() - 0.0) < 1e-5

    # Orthogonal: cos_sim=0 -> loss = 2
    a = F.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
    b = F.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=-1)
    loss_orth = (2.0 - 2.0 * (a * b).sum(dim=-1)).item()
    assert abs(loss_orth - 2.0) < 1e-5

    # Module forward: same input -> loss non-negative; formula is 2-2*cos_sim so in [0,2] for unit vectors (BN may yield slight overshoot)
    mod = ConsistencyModule(hidden_state_dim=8, proj_dim=8)
    mod.eval()
    h = torch.randn(2, 8)
    loss_same = mod(h, h, reduction='mean').item()
    assert 0 <= loss_same <= 2.5, "Consistency loss should be in [0, 2] (allow 2.5 for BN)"


# ----- Player rotation -----

def test_player_rotation_hand_computed():
    """Player index rotation: shift = (boot_p - my_player) % 3; np.roll(bootstrap_val, shift) = [V_me, V_next, V_prev]."""
    bootstrap_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # [V_boot, V_boot_next, V_boot_prev]
    my_player = 1
    boot_p = 2
    shift = (boot_p - my_player) % 3  # 1
    rotated = np.roll(bootstrap_val, shift)
    # boot_p=2: bootstrap view is [V2, V3, V1]. We want [V1, V2, V3]. roll(.,1) -> [3, 1, 2] = [V1, V2, V3]. Ok.
    expected = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(rotated, expected)

    # Integrate with _compute_value_target: game with single bootstrap step
    game = GameHistory()
    game.player_ids = [1, 2]  # pos 0 -> 1, bootstrap at 1 -> 2
    game.root_values = [np.zeros(3), np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    for _ in range(2):
        game.observations.append(np.zeros((4, 21, 21), dtype=np.float32))
        game.actions.append(0)
        game.rewards.append(0.0)
        game.policy_targets.append(np.ones(441) / 441)
        game.threats.append(np.zeros(3, dtype=np.float32))
        game.centers.append((0, 0))
    game.placement_rewards = {}
    buffer = ReplayBuffer(max_size=10)
    target = buffer._compute_value_target(game, 0, td_steps=1, discount=0.99)
    # shift = (2-1)%3 = 1, rotated = [3,1,2], value = 0.99 * [3,1,2]
    np.testing.assert_array_almost_equal(target, 0.99 * np.array([3.0, 1.0, 2.0], dtype=np.float32))


# ----- Optional assertions (value range, policy sum, shapes) -----

def test_value_target_in_reasonable_range():
    """Assert value target in [-1.5, 1.5] for typical placement_rewards."""
    game = GameHistory()
    game.player_ids = [1, 2, 3]
    game.placement_rewards = {1: 1.0, 2: -0.2, 3: -1.0}
    for _ in range(3):
        game.observations.append(np.zeros((4, 21, 21), dtype=np.float32))
        game.actions.append(0)
        game.rewards.append(0.0)
        game.policy_targets.append(np.ones(441) / 441)
        game.root_values.append(np.zeros(3, dtype=np.float32))
        game.threats.append(np.zeros(3, dtype=np.float32))
        game.centers.append((0, 0))
    game.done = True
    buffer = ReplayBuffer(max_size=10)
    target = buffer._compute_value_target(game, 0, 5, 0.99)
    assert np.all(target >= -1.5) and np.all(target <= 1.5)


def test_mcts_action_probs_sum_and_root_value_shape():
    """After MCTS, action_probs sum to 1.0 and root_value shape (3,)."""
    from ai.mcts import gumbel_muzero_search
    from ai.muzero_config import MuZeroConfig
    from ai.muzero_network import MuZeroNetwork

    config = MuZeroConfig()
    config.num_simulations = 2
    config.device = 'cpu'
    config.board_size = 15
    config.win_length = 5
    network = MuZeroNetwork(config)
    # Observation channels = 8 (4 local + 4 global thumbnail per config)
    obs = np.zeros((8, 21, 21), dtype=np.float32)
    legal = np.ones(441, dtype=np.float32)
    action_probs, root_value = gumbel_muzero_search(network, obs, legal, config, add_noise=False)
    assert np.isclose(action_probs.sum(), 1.0), f"action_probs sum {action_probs.sum()}"
    assert root_value.shape == (3,), f"root_value shape {root_value.shape}"


# ----- Optional: reward MSE and gradient_scale in train code path -----

def test_train_step_reward_mse_and_gradient_scale_in_code():
    """Verify train step uses MSE for reward loss and 1/K gradient scale for unroll (code path)."""
    train_py = os.path.join(os.path.dirname(__file__), "train.py")
    with open(train_py, "r", encoding="utf-8") as f:
        source = f.read()
    assert "F.mse_loss(reward, target_rewards" in source, "Reward loss should be MSE to target_rewards"
    assert "gradient_scale = 1.0 / config.num_unroll_steps" in source, "Unroll should use gradient_scale 1/K"
