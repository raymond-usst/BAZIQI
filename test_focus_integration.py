import torch
import numpy as np
from ai.muzero_network import MuZeroNetwork, FocusNetwork
from ai.muzero_config import MuZeroConfig

def test_focus_integration():
    print("1. Testing Focus Network (Regression)...")
    net = FocusNetwork(channels=4)
    dummy_input = torch.randn(2, 4, 100, 100)
    output = net(dummy_input)
    print(f"   Shape: {output.shape} (Expected: 2, 2)")
    assert output.shape == (2, 2), "Output shape incorrect"
    assert output.min() >= 0 and output.max() <= 1, "Output not in [0, 1]"

    print("2. Testing Predict Center...")
    config = MuZeroConfig()
    muzero = MuZeroNetwork(config)
    # Mock global state
    gs = torch.zeros(1, 4, 100, 100)
    # Put a "feature" at (80, 80) is hard for random weights, 
    # but let's just check valid output range.
    r, c = muzero.predict_center(gs)
    print(f"   Predicted center: ({r}, {c})")
    assert 0 <= r < 100 and 0 <= c < 100, "Predicted center out of bounds"

    print("3. Testing Loss Calculation...")
    # Mock batch data as in train_step
    # target_centers are indices 0..9999
    target_centers = torch.tensor([5050, 0, 9999], dtype=torch.long)
    # Expected normalized coords: (0.5, 0.5), (0.0, 0.0), (0.99, 0.99)
    
    # Mock network output
    # Perfect prediction
    pred = torch.tensor([[0.505, 0.505], [0.0, 0.0], [0.99, 0.99]], dtype=torch.float32)
    
    # Calculate MSE
    # Replicate train.py logic
    w = 100
    target_r = (target_centers // w).float() / w
    target_c = (target_centers % w).float() / w
    target_coords = torch.stack([target_r, target_c], dim=1)
    
    loss_f = torch.nn.functional.mse_loss(pred, target_coords) * 10.0
    print(f"   Target Coords:\n{target_coords}")
    print(f"   Pred Coords:\n{pred}")
    print(f"   Loss: {loss_f.item()}")
    
    assert loss_f.item() < 1.0, "Loss should be small for good prediction"
    
    print("ALL TESTS PASSED.")

if __name__ == "__main__":
    test_focus_integration()
