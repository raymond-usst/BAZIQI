import numpy as np
from ai.game_env import EightInARowEnv

def test_smart_center():
    env = EightInARowEnv()
    board_size = env.BOARD_SIZE
    view_size = 21
    half = view_size // 2
    
    # 1. Simulate a clogged center
    # Fill a 21x21 block in the absolute center of the board
    center_r, center_c = board_size // 2, board_size // 2
    
    # Use direct manipulation to avoid Game Over from win check
    # We just want to test the view logic, not game rules
    print("Filling center block...")
    pid = 1
    for r in range(center_r - half, center_r + half + 1):
        for c in range(center_c - half, center_c + half + 1):
            env.board[r, c] = pid
            pid = (pid % 3) + 1 # Rotate players
            
            # Manually update bounding box as step() would
            env._min_r = min(env._min_r, r)
            env._max_r = max(env._max_r, r)
            env._min_c = min(env._min_c, c)
            env._max_c = max(env._max_c, c)
            
    # At this point, the geometric center (50, 50) is full.
    # get_observation() should NOT return (50, 50) or fail.
    # It should return a shifted center.
    
    obs, (cr, cc) = env.get_observation(view_size)
    print(f"Geometric Center would be: {env.get_center()}")
    print(f"Smart Center returned: {(cr, cc)}")
    
    # Check if this new center has legal moves
    legal = env.legal_moves_local(cr, cc, view_size)
    print(f"Legal moves at smart center: {len(legal)}")
    
    if len(legal) > 0 and (cr != center_r or cc != center_c):
        print("SUCCESS: Smart center found valid moves and shifted away from clogged center.")
    else:
        print("FAILURE: Smart center failed.")
        print(f"Center diff: {cr-center_r}, {cc-center_c}")
        print(f"Legal count: {len(legal)}")

if __name__ == "__main__":
    test_smart_center()
