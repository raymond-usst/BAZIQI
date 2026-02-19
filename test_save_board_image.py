"""Quick test: save a self-play style board to img folder."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai.board_render import board_to_image_path

def main():
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, 'test_quick.png')

    # Synthetic board: 21x21, a few moves (0=empty, 1=red, 2=green, 3=blue)
    board = np.zeros((21, 21), dtype=np.int8)
    board[10, 8:12] = 1   # red line
    board[9:12, 10] = 2   # green line
    board[8, 9] = 3
    board[9, 9] = 3
    board[10, 9] = 3
    board[11, 9] = 3
    board[12, 9] = 3     # blue line

    ok = board_to_image_path(board, path, cell_px=6)
    if ok:
        print(f"OK: saved board image to {path}")
        print(f"    File exists: {os.path.exists(path)}")
    else:
        print("FAIL: could not save (install matplotlib: pip install matplotlib)")
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
