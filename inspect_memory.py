
import os
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def inspect_memory(path):
    print(f"Inspecting MemoryBank: {path}")
    if not os.path.exists(path):
        print("  File not found.")
        return

    try:
        data = torch.load(path, map_location='cpu')
        keys = data['keys']
        values = data['values']
        count = data['count']

        print(f"  Count: {count}")
        
        # Check NaNs
        k_nan = torch.isnan(keys).sum().item()
        v_nan = torch.isnan(values).sum().item()
        k_inf = torch.isinf(keys).sum().item()
        v_inf = torch.isinf(values).sum().item()
        
        print(f"  Keys NaNs: {k_nan}, Infs: {k_inf}")
        print(f"  Values NaNs: {v_nan}, Infs: {v_inf}")

        if count > 0:
            active_keys = keys[:count]
            active_values = values[:count]
            
            print(f"  Keys Abs Mean: {active_keys.abs().mean().item():.4f}, Max: {active_keys.abs().max().item():.4f}")
            print(f"  Values Abs Mean: {active_values.abs().mean().item():.4f}, Max: {active_values.abs().max().item():.4f}")
            
            # Check for near-zero variance in keys (could cause normalization issues)
            key_std = active_keys.std(dim=0).mean().item()
            print(f"  Keys Mean Std: {key_std:.4f}")

    except Exception as e:
         print(f"  Error loading memory: {e}")

if __name__ == '__main__':
    inspect_memory(r'checkpoints_async\shared_memory.pt')
