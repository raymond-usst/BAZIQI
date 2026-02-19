
import sys
import os
import unittest
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.league import LeagueManager
from ai.muzero_config import MuZeroConfig

class TestLeagueManager(unittest.TestCase):
    def setUp(self):
        self.config = MuZeroConfig()
        # Mock checkpoint dir to avoid creating files in real dirs?
        self.config.checkpoint_dir = "test_checkpoints" 
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        self.league = LeagueManager(self.config, league_file="test_league.json")

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.league.league_file):
            os.remove(self.league.league_file)
        if os.path.exists(self.config.checkpoint_dir):
            try:
                os.rmdir(self.config.checkpoint_dir)
            except: pass

    def test_elo_update_equal_rating(self):
        # Case 1: Equal rating (1200 vs 1200), A wins.
        # Exp = 0.5. K=32.
        # A Gain = 32 * (1 - 0.5) = 16
        # B Loss = 32 * (0 - 0.5) = -16
        r_a, r_b = self.league.update_elo(1200, 1200, 1.0)
        self.assertAlmostEqual(r_a, 1216.0)
        self.assertAlmostEqual(r_b, 1184.0)
        
    def test_elo_update_stronger_wins(self):
        # Case 2: 1600 vs 1200. A wins.
        # Delta = 400. Exp_A = 1 / (1 + 10^(-400/400)) = 1 / 1.1 = 0.909090...
        ra = 1600
        rb = 1200
        exp_a = 1.0 / 1.1
        k = 32
        expected_new_a = ra + k * (1.0 - exp_a)
        expected_new_b = rb + k * (0.0 - (1.0 - exp_a))
        
        new_a, new_b = self.league.update_elo(ra, rb, 1.0)
        
        self.assertAlmostEqual(new_a, expected_new_a)
        self.assertAlmostEqual(new_b, expected_new_b)
        
        # Approximate check
        self.assertTrue(1602.0 < new_a < 1603.5)
        
    def test_elo_update_upset(self):
        # Case 3: 1600 vs 1200. A loses (0.0).
        ra = 1600
        rb = 1200
        exp_a = 1.0 / 1.1 # ~0.909
        
        new_a, new_b = self.league.update_elo(ra, rb, 0.0)
        
        # A should lose a lot
        # Loss = 32 * (0 - 0.909) = -29.09
        self.assertTrue(new_a < 1572.0)
        
        # B should gain a lot
        self.assertTrue(new_b > 1228.0)

    def test_draw(self):
        # Case: 1200 vs 1200 draw
        r_a, r_b = self.league.update_elo(1200, 1200, 0.5)
        self.assertAlmostEqual(r_a, 1200.0)
        self.assertAlmostEqual(r_b, 1200.0)
        
        # Case: 1600 vs 1200 draw
        # A is expected to win (0.909). Scored 0.5. Underperformed. Should lose Elo.
        # B is expected to lose (0.091). Scored 0.5. Overperformed. Should gain Elo.
        new_a, new_b = self.league.update_elo(1600, 1200, 0.5)
        self.assertTrue(new_a < 1600)
        self.assertTrue(new_b > 1200)

if __name__ == '__main__':
    unittest.main()
