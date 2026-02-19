
import pickle
import numpy as np
import os
import sys

# Import ReplayBuffer to ensure class is available for unpickling
sys.path.append(os.getcwd())
try:
    from ai.replay_buffer import ReplayBuffer, GameHistory
except ImportError:
    print("Could not import ReplayBuffer. Make sure you are in the project root.")
    # Define dummy class if needed, but better to fail
    # class ReplayBuffer: pass

def sanitize():
    path = "checkpoints_async/replay_buffer.pkl"
    if not os.path.exists(path):
        print(f"No buffer found at {path}")
        return

    print(f"Loading {path}...")
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if hasattr(obj, 'buffer'):
        buffer_list = obj.buffer
    elif isinstance(obj, dict) and 'buffer' in obj:
        buffer_list = obj['buffer']
    else:
        print(f"Unknown buffer format: {type(obj)}")
        return

    print(f"Loaded {len(buffer_list)} games. Scanning for corruption...")
    
    clean_buffer = []
    poison_count = 0
    
    for i, game in enumerate(buffer_list):
        is_poison = False
        
        # Check observations, rewards, policies, root_values
        # obs: list of np.ndarray
        # rewards: list of float
        # policy: list of list/array
        # root_values: list of float
        
        # Check root values
        if any(np.isnan(v) or np.isinf(v) for v in game.root_values):
            is_poison = True
            print(f"Game {i}: NaN in root_values")
        
        # Check rewards
        if not is_poison and any(np.isnan(r) or np.isinf(r) for r in game.rewards):
            is_poison = True
            print(f"Game {i}: NaN in rewards")
            
        # Check policies
        if not is_poison:
            for p in game.policy_targets: 
                 if np.sum(p) > 0 and (np.any(np.isnan(p)) or np.any(np.isinf(p))):
                     is_poison = True
                     print(f"Game {i}: NaN in policies")
                     break

        # Check observations (random sample to check speed?)
        # Or check all? Obs are 8x21x21.
        if not is_poison:
             for obs in game.observations:
                 if np.isnan(obs).any() or np.isinf(obs).any():
                     is_poison = True
                     print(f"Game {i}: NaN in observations")
                     break

        if is_poison:
            poison_count += 1
        else:
            clean_buffer.append(game)

    print(f"Scan complete.")
    print(f"Total Games: {len(buffer_list)}")
    print(f"Poisoned Games: {poison_count}")
    print(f"Clean Games: {len(clean_buffer)}")

    if poison_count > 0:
        if hasattr(obj, 'buffer'):
            obj.buffer = clean_buffer
        elif isinstance(obj, dict) and 'buffer' in obj:
            obj['buffer'] = clean_buffer
            
        out_path = "checkpoints_async/replay_buffer_clean.pkl"
        print(f"Saving clean buffer to {out_path}...")
        with open(out_path, "wb") as f:
            pickle.dump(obj, f)
        print("Done. Please replace replay_buffer.pkl with replay_buffer_clean.pkl")
    else:
        print("Buffer is clean! No action needed.")

if __name__ == "__main__":
    sanitize()
