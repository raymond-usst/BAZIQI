"""
Recovery script v3 — extract 848 GameHistory objects from metastack frame 2.

From v2 we learned the stack structure:
  Frame 0: [empty dict]           — the outer result dict being built
  Frame 1: ['buffer', empty list] — key + list (APPENDS not yet called)
  Frame 2: [848 GameHistory]      — pending items for the list!
  Frame 3: ['observations', []]   — partial GameHistory being built (truncated)
  
Main stack: 10 ndarrays — the observations list of the in-progress game

Solution: grab all GameHistory objects from Frame 2.
"""

import pickle
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def recover_replay_buffer(input_path, output_path):
    print(f"=== Replay Buffer Recovery v3 ===")
    file_size = os.path.getsize(input_path)
    print(f"Input: {file_size:,} bytes ({file_size/1024/1024/1024:.2f} GB)")
    print()
    print("Parsing with pure Python _Unpickler...")
    
    with open(input_path, 'rb') as f:
        u = pickle._Unpickler(f)
        try:
            result = u.load()
            print("File is NOT corrupted!")
            if isinstance(result, dict) and 'buffer' in result:
                print(f"Buffer: {len(result['buffer'])} games")
            return
        except Exception as e:
            print(f"Truncation at byte {f.tell():,}: {e}")
    
    print()
    print("=== Inspecting unpickler state ===")
    
    # Collect ALL GameHistory objects from all frames
    from ai.replay_buffer import GameHistory
    recovered = []
    
    # Check metastack frames
    for fi, frame in enumerate(u.metastack):
        count_in_frame = 0
        for item in frame:
            try:
                if isinstance(item, GameHistory) and hasattr(item, 'observations') and len(item.observations) > 0:
                    recovered.append(item)
                    count_in_frame += 1
            except Exception:
                pass  # skip broken objects
        
        if count_in_frame > 0:
            print(f"  Frame {fi}: {count_in_frame} valid GameHistory objects")
    
    # Check main stack too
    for item in u.stack:
        try:
            if isinstance(item, GameHistory) and hasattr(item, 'observations') and len(item.observations) > 0:
                recovered.append(item)
        except Exception:
            pass
    
    print(f"\nTotal recovered: {len(recovered)} GameHistory objects")
    
    if not recovered:
        print("No data to save.")
        return
    
    # Stats
    total_pos = sum(len(g) for g in recovered)
    avg_len = total_pos / len(recovered)
    print(f"Total positions: {total_pos:,}")
    print(f"Average game length: {avg_len:.1f}")
    
    # Preview first few games
    for i in range(min(3, len(recovered))):
        g = recovered[i]
        print(f"  Game {i}: {len(g)} steps, winner={g.winner}, actions={len(g.actions)}")
    
    # Save
    from ai.replay_buffer import ReplayBuffer
    buf = ReplayBuffer(max_size=50000)
    for g in recovered:
        buf.save_game(g)
    
    buf.save(output_path)
    out_size = os.path.getsize(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Size: {out_size:,} bytes ({out_size/1024/1024:.1f} MB)")
    return len(recovered)


if __name__ == '__main__':
    input_path = r"d:\code\trifeet\checkpoints_async\replay_buffer.pkl"
    output_path = r"d:\code\trifeet\checkpoints_async\replay_buffer_recovered.pkl"
    
    start = time.time()
    count = recover_replay_buffer(input_path, output_path)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s")
    
    if count:
        print(f"\n*** SUCCESS: {count} games recovered! ***")
    else:
        print("\n*** No games recovered ***")
