
import os
import torch
import torch.nn.functional as F
import sys
import pickle

# Ensure we can import ai modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.muzero_config import MuZeroConfig
from ai.muzero_network import MuZeroNetwork
from ai.replay_buffer import ReplayBuffer
from ai.engram import MemoryBank
from ai.train import train_step

def debug_nan():
    print("Starting debug_nan_step...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = MuZeroConfig()
    
    # 1. Load Model
    model = MuZeroNetwork(config).to(device)
    checkpoint_path = r'checkpoints_async/latest.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint step: {checkpoint.get('step', '?')}")
    else:
        print("Loaded weights-only checkpoint (no optimizer/scaler state).")
        model.load_state_dict(checkpoint)
        # Optimizer/Scaler will be fresh, which is fine for debugging step
    model.train()
    
    # 2. Load Optimizer (to check state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 3. Load Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        s = scaler.get_scale()
        print(f"Scaler scale from checkpoint: {s}")
        if s <= 0.0:
            print("Resetting scaler to default (65536).")
            scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 4. Load Memory Bank
    memory_bank = None
    if config.use_engram:
        memory_bank = MemoryBank(config.memory_capacity, config.hidden_state_dim, config.hidden_state_dim)
        memory_path = r'checkpoints_async/shared_memory.pt'
        if os.path.exists(memory_path):
             print(f"Loading memory: {memory_path}")
             mem_state = torch.load(memory_path, map_location='cpu')
             memory_bank.load_state_dict(mem_state)
             # Manually move to device
             memory_bank.keys = memory_bank.keys.to(device)
             memory_bank.values = memory_bank.values.to(device)
             memory_bank.priorities = memory_bank.priorities.to(device)

    # 5. Load Replay Buffer & Sample
    buffer_path = r'checkpoints_async/replay_buffer.pkl'
    if not os.path.exists(buffer_path):
        print("Buffer not found.")
        return
        
    print(f"Loading buffer: {buffer_path}")
    replay_buffer = ReplayBuffer(config.replay_buffer_size)
    replay_buffer.load(buffer_path)
    
    if len(replay_buffer) < config.batch_size:
        print(f"Buffer too small: {len(replay_buffer)}")
        return
        
    print("Sampling batch...")
    batch = replay_buffer.sample_batch(config.batch_size, config.num_unroll_steps, config.td_steps, config.discount, config.policy_size)
    
    print("Running train_step...")
    # We use the train_step from ai.train which has our debug prints
    info = train_step(model, optimizer, scaler, batch, config, device, memory_bank)
    
    print("Train step finished.")
    print(f"Loss: {info['loss']}")
    print(f"Loss details: {info}")

if __name__ == "__main__":
    try:
        debug_nan()
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
