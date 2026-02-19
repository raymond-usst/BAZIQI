
import os
import torch
import torch.nn as nn
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.muzero_config import MuZeroConfig
from ai.muzero_network import MuZeroNetwork

def debug_focus():
    print("Starting debug_focus_net...")
    # Force CPU to match Actors
    device = torch.device('cpu')
    print(f"Device: {device}")

    config = MuZeroConfig()
    model = MuZeroNetwork(config).to(device)
    
    checkpoint_path = r'checkpoints_async/latest.pt'
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found.")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    # Run Sanitization Logic (copy from train_async.py)
    print("Running sanitization check...")
    nan_fixed = 0
    for name, param in list(model.named_parameters()) + list(model.named_buffers()):
        is_var = 'running_var' in name or 'var' in name
        
        if torch.isnan(param).any() or torch.isinf(param).any() or (is_var and (param < 0).any()):
            print(f"FOUND corruption in: {name}")
            if torch.isnan(param).any(): print("  - Contains NaN")
            if torch.isinf(param).any(): print("  - Contains Inf")
            if is_var and (param < 0).any(): print("  - Contains Negative Var")
            
            with torch.no_grad():
                if is_var:
                    param[torch.isnan(param)] = 1.0
                    param[torch.isinf(param)] = 1.0
                    param[param < 0] = 1.0
                else:
                    param[torch.isnan(param)] = 0.0
                    param[torch.isinf(param)] = 0.0
            nan_fixed += 1
            
    print(f"Sanitized {nan_fixed} tensors.")

    # Test Focus Network
    print("Testing Focus Network...")
    model.eval() # Actor usage
    dummy_input = torch.rand(1, 4, 100, 100).to(device)
    print(f"Input mean: {dummy_input.mean():.4f}, std: {dummy_input.std():.4f}")
    
    with torch.no_grad():
        output = model.focus_net(dummy_input)
    
    print(f"Output: {output}")

    print("Inspecting Focus Network buffers (regardless of output):")
    for name, buf in model.focus_net.named_buffers():
        if 'running_var' in name:
            print(f"{name}: min={buf.min()}, max={buf.max()}, mean={buf.mean()}")
            if (buf < 0).any(): print(f"  !!! NEGATIVE VARIANCE !!!")
    
    if torch.isnan(output).any() or torch.isinf(output).any():
        print("!!! Focus Network produced NaN/Inf !!!")
        print("Inspecting Focus Network weights:")
        for name, param in model.focus_net.named_parameters():
            print(f"{name}: min={param.min()}, max={param.max()}, mean={param.mean()}")
        for name, buf in model.focus_net.named_buffers():
            print(f"{name}: min={buf.min()}, max={buf.max()}, mean={buf.mean()}")
    else:
        print("Focus Network seems fine on dummy input.")

if __name__ == "__main__":
    debug_focus()
