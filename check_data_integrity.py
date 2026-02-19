
import os
import torch
import pickle
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_replay_buffer(path):
    print(f"Checking ReplayBuffer: {path}")
    if not os.path.exists(path):
        print("  File not found.")
        return

    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        buffer = data['buffer']
        print(f"  Total games: {len(buffer)}")
        
        nan_games = 0
        for i, game in enumerate(buffer):
            has_nan = False
            # check obs
            for obs in game.observations:
                if np.isnan(obs).any() or np.isinf(obs).any():
                    has_nan = True; break
            
            if not has_nan:
                for r in game.rewards:
                    if np.isnan(r) or np.isinf(r): has_nan = True; break
            
            if not has_nan:
                for v in game.root_values:
                    if np.isnan(v) or np.isinf(v): has_nan = True; break
            
            if has_nan:
                nan_games += 1
                if nan_games <= 5:
                    print(f"  Game {i} contains NaN/Inf!")

        print(f"  Corrupted games: {nan_games} / {len(buffer)}")

    except Exception as e:
        print(f"  Error loading buffer: {e}")

def check_memory_bank(path):
    print(f"Checking MemoryBank: {path}")
    if not os.path.exists(path):
        print("  File not found.")
        return
    
    try:
        data = torch.load(path, map_location='cpu')
        keys = data['keys']
        values = data['values']
        
        k_nan = torch.isnan(keys).sum().item()
        v_nan = torch.isnan(values).sum().item()
        k_inf = torch.isinf(keys).sum().item()
        v_inf = torch.isinf(values).sum().item()
        
        print(f"  Count: {data['count']}")
        print(f"  Keys NaNs: {k_nan}, Infs: {k_inf}")
        print(f"  Values NaNs: {v_nan}, Infs: {v_inf}")
        
    except Exception as e:
         print(f"  Error loading memory: {e}")

if __name__ == '__main__':
    check_replay_buffer(r'checkpoints_async\replay_buffer.pkl')
    check_memory_bank(r'checkpoints_async\shared_memory.pt')
