"""Smoke test for Advanced MuZero Architecture."""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.muzero_config import MuZeroConfig
from ai.muzero_network import MuZeroNetwork
from ai.replay_buffer import ReplayBuffer, GameHistory
from ai.mcts import gumbel_muzero_search
from ai.engram import MemoryBank
from ai.train import train_step

def test_architecture():
    print("=" * 60)
    print("  Advanced Architecture Smoke Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Config & Network
    config = MuZeroConfig()
    config.d_model = 64        # Smaller for test
    config.n_layers = 2
    config.n_heads = 4
    config.hidden_state_dim = 32
    config.fc_hidden = 64
    config.batch_size = 4
    config.device = str(device)
    
    print("\n[1] Initializing Network (Transformer + Engram + Consistency)...")
    network = MuZeroNetwork(config).to(device)
    print("    Success.")

    # 2. Inference
    print("\n[2] Testing Inference...")
    B = 2
    obs = torch.randn(B, 8, 21, 21).to(device)
    
    # Mock memory retrieval
    mem_keys = torch.randn(B, config.memory_top_k, 32).to(device)
    mem_vals = torch.randn(B, config.memory_top_k, 32).to(device)
    
    # Initial
    with torch.no_grad():
        h, p, v = network.initial_inference(obs, mem_keys, mem_vals)
    
    # Verify standard outputs
    print(f"    Initial Inference: h={h.shape}, p={p.shape}, v={v.shape}")
    assert h.shape == (B, 32)
    assert p.shape == (B, 441)
    assert v.shape == (B, 3)

    # Verify Aux outputs via direct prediction call
    with torch.no_grad():
        policy, value, threat, opp, heatmap = network.prediction(h)
    
    print(f"    Aux: threat={threat.shape}, opp={opp.shape}, heatmap={heatmap.shape}")
    
    # Assert aux shapes
    assert threat.shape == (B, 3)
    assert opp.shape == (B, config.policy_size)
    assert heatmap.shape == (B, 1, 21, 21)

    # Recurrent
    action = torch.randint(0, 441, (B,)).to(device)
    h_next, r, p2, v2 = network.recurrent_inference(h, action, mem_keys, mem_vals)
    print(f"    Recurrent Inference: h_next={h_next.shape}, r={r.shape}")
    assert h_next.shape == (B, 32)
    
    # Consistency
    h_proj = network.project(h_next)
    h_pred = network.predict_projection(h_proj)
    loss_c = network.consistency(h_next, h_next.detach()) # dummy
    print(f"    Consistency Loss: {loss_c.item():.4f}")

    # 3. MCTS
    print("\n[3] Testing Gumbel MuZero Search...")
    obs_np = np.random.randn(8, 21, 21).astype(np.float32)
    legal_mask = np.ones(441, dtype=np.float32)
    
    probs, root_value, _root = gumbel_muzero_search(network, obs_np, legal_mask, config, add_noise=True)
    print(f"    Action Probs: {probs.shape}, Sum={probs.sum():.2f}, RootValue shape={root_value.shape}")
    assert probs.shape == (441,)
    assert abs(probs.sum() - 1.0) < 1e-3
    assert root_value.shape == (3,) # Vector value

    # 4. Replay Buffer & Training
    print("\n[4] Testing Training Step...")
    buffer = ReplayBuffer(max_size=100)
    for _ in range(5):
        game = GameHistory()
        for i in range(10):
            game.store(
                np.random.randint(0, 441),
                0.0,
                np.ones(441, dtype=np.float32) / 441,
                np.zeros(3, dtype=np.float32), # Vector root value
                np.zeros(3, dtype=np.float32), # Threats
                1,
                (50, 50)
            )
        buffer.save_game(game)
    
    batch = buffer.sample_batch(B, 5, 5, 0.99, 441)
    assert 'next_observations' in batch
    print(f"    Batch sampled. Next obs shape: {batch['next_observations'].shape}")
    
    memory_bank = MemoryBank(100, 32, 32)
    # Write some dummy keys
    memory_bank.write(torch.randn(10, 32), torch.randn(10, 32))
    
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=False) # Disable amp for smoke test sim? or usage
    if device.type == 'cuda':
         scaler = torch.amp.GradScaler('cuda')

    loss_dict = train_step(network, optimizer, scaler, batch, config, device, memory_bank)
    print(f"    Train Step Loss: {loss_dict['total']:.4f}")
    print(f"    Consistency Loss: {loss_dict['consistency']:.4f}")
    
    print("\n✅ Validated: Transformer + Gumbel + Consistency + Engram")

    # 5. GameEnv
    print("\n[5] Testing GameEnv Observation...")
    from ai.game_env import EightInARowEnv
    env = EightInARowEnv()
    obs, _ = env.get_observation()
    print(f"    Observation Shape: {obs.shape}")
    assert obs.shape == (8, 21, 21)
    print("    Success.")

if __name__ == '__main__':
    try:
        test_architecture()
    except Exception as e:
        import traceback
        with open('crash.txt', 'w') as f:
            traceback.print_exc(file=f)
        raise
