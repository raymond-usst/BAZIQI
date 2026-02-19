
import torch
import sys
import os

def inspect_crash(dump_path):
    print(f"Loading crash dump: {dump_path}")
    if not os.path.exists(dump_path):
        print("File not found.")
        return

    data = torch.load(dump_path, map_location='cpu')
    
    # 1. Inspect Input
    gs = data['global_state']
    print(f"Global State: shape={gs.shape}, dtype={gs.dtype}")
    print(f"  Min={gs.min()}, Max={gs.max()}, Mean={gs.mean()}")
    if torch.isnan(gs).any(): print("  !!! GLOBAL STATE CONTAINS NaN !!!")
    if torch.isinf(gs).any(): print("  !!! GLOBAL STATE CONTAINS Inf !!!")
    
    # 2. Inspect Weights
    print("Inspecting Network State Dict:")
    state_dict = data['network_state']
    nan_params = []
    
    for name, param in state_dict.items():
        if torch.isnan(param).any():
            nan_params.append(name)
            print(f"  NaN in: {name}")
        if 'running_var' in name and (param < 0).any():
             print(f"  Negative Variance in: {name} (min={param.min()})")

    if not nan_params:
        print("  No NaN weights found.")
    else:
        print(f"  Found {len(nan_params)} corrupted weights.")

    # 3. Inspect Focus Buffers
    print("Inspecting Focus Net Buffers:")
    buffers = data.get('focus_net_buffers', {})
    for name, buf in buffers.items():
        print(f"  {name}: min={buf.min()}, max={buf.max()}")
        if torch.isnan(buf).any(): print("    !!! NaN !!!")
        if (buf < 0).any(): print("    !!! NEGATIVE !!!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Try to find one
        files = [f for f in os.listdir('.') if f.startswith('debug_crash_actor_') and f.endswith('.pt')]
        if files:
            path = files[0]
        else:
            print("No crash file found.")
            sys.exit(1)
            
    inspect_crash(path)
