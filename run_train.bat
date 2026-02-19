@echo off
echo Starting Async Training (Resuming)...
python ai/train_async.py --resume --actors 8
pause
