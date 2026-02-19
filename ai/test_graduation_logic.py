
import sys
import os
import numpy as np
import unittest

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.muzero_config import MuZeroConfig
from ai.curriculum import CurriculumManager

class TestGraduationLogic(unittest.TestCase):
    def setUp(self):
        self.config = MuZeroConfig()
        self.cm = CurriculumManager(self.config)

    def test_dynamic_game_count(self):
        # Stage 1: 15x15 -> 225. Ratio=1. N=base_games (default 10000)
        s1 = self.cm.stages[0]
        self.assertEqual(s1.board_size, 15)
        self.assertEqual(s1.min_games, 10000)
        
        # Stage 2: 30x30 -> 900. Ratio=4. N=10000 * 4^1.5 = 80000
        s2 = self.cm.stages[1]
        self.assertEqual(s2.board_size, 30)
        self.assertAlmostEqual(s2.min_games, 80000, delta=100)
        
        # Stage 4: 100x100. Formula gives ~2.96M with base 10k; capped at 2M so >= 50x50
        s4 = self.cm.stages[3]
        self.assertEqual(s4.board_size, 100)
        self.assertEqual(s4.min_games, 2_000_000)

    def test_wilson_score(self):
        # Test helper directly
        # 100 wins / 100 games -> LCB should be high (near 0.94)
        lcb_perfect = self.cm._wilson_score_lower_bound(100, 100)
        self.assertTrue(lcb_perfect > 0.90)
        
        # 55 wins / 100 games -> 0.55 avg. 
        # Error margin at 99% conf for n=100 is approx 1.29/sqrt(100) * ... ~0.12
        # LCB should be around 0.43 -> Should FAIL threshold 0.55
        lcb_marginal = self.cm._wilson_score_lower_bound(55, 100)
        self.assertTrue(lcb_marginal < 0.50)
        
        # 600 wins / 1000 games -> 0.60 avg.
        # n=1000, error margin smaller. LCB should be > 0.55
        lcb_good = self.cm._wilson_score_lower_bound(600, 1000)
        self.assertTrue(lcb_good > 0.55)
        
    def test_loss_convergence(self):
        # 1. Still decreasing (Not converged)
        # Window 1 (recent): 0.5. Window 2 (prev): 0.6. Diff: 0.1 > epsilon.
        self.cm.loss_buffer = [0.8] * 100 + [0.7] * 100 + [0.6] * 100 + [0.5] * 100
        # Use window=100.
        self.assertFalse(self.cm._check_loss_convergence(window=100, epsilon=0.01))
        
        # 2. Steady Decrease (Linear)

        self.cm.loss_buffer = [1.0 - i*0.001 for i in range(1000)]
        # Deriv = 0.001 approx. 
        # current mean approx 0.0. prev mean approx 0.5. diff 0.5. False.
        self.assertFalse(self.cm._check_loss_convergence(window=100))
        
        # 3. Plateau
        self.cm.loss_buffer = [0.1] * 1000
        self.assertTrue(self.cm._check_loss_convergence(window=100, epsilon=1e-4))

    def test_full_graduation_logic(self):
        # Manually satisfy all conditions
        
        # 1. Min Games
        self.cm.games_in_stage = 20001
        
        # 2. Win Rate (Wilson)
        # Need LCB > 0.60 (Stage 1 threshold)
        # Try 70% win rate over 1000 games
        for _ in range(1000):
            self.cm.win_rate_buffer.append(1.0 if np.random.rand() < 0.7 else 0.0)
            
        # 3. Loss Convergence
        self.cm.loss_buffer = [0.1] * 2000
        
        self.assertTrue(self.cm.check_graduation())
        
        # Fail one condition: Games
        self.cm.games_in_stage = 100
        self.assertFalse(self.cm.check_graduation())

    def test_baseline_wr_blocks_graduation(self):
        # LCB above threshold but below baseline 0.50 should not graduate
        self.cm.games_in_stage = 20001
        self.cm.win_rate_buffer = [0.52] * 1000  # ~52% -> LCB might be just above threshold but < 0.50 with small n
        # Actually 520 wins/1000 -> LCB ~0.49. So LCB < 0.50.
        self.cm.loss_buffer = [0.1] * 2000
        self.assertFalse(self.cm.check_graduation())

    def test_state_dict_roundtrip(self):
        self.cm.games_in_stage = 100
        self.cm.win_rate_buffer = [1.0] * 50
        self.cm.stage_start_elo = 1234.5
        sd = self.cm.state_dict()
        self.assertIn('stage_start_elo', sd)
        self.assertEqual(sd['stage_start_elo'], 1234.5)
        cm2 = CurriculumManager(self.config)
        cm2.load_state_dict(sd)
        self.assertEqual(cm2.games_in_stage, 100)
        self.assertEqual(cm2.stage_start_elo, 1234.5)

if __name__ == '__main__':
    unittest.main()
