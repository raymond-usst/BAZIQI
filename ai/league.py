"""League of historical opponents with Elo; sampling and eviction.

Maintainability: load/save retry, empty pool handling — see ai/MAINTENANCE.md.
"""

import os
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from ai.log_utils import get_logger

_log = get_logger(__name__)

@dataclass
class LeagueOpponent:
    checkpoint_path: str
    elo: float
    step: int
    added_at: float  # timestamp

class LeagueManager:
    """
    Manages a pool of historical opponents and their Elo ratings.

    Elasticity: league_max_snapshots caps the opponent pool size; oldest entries
    are evicted when over limit. When the pool is empty, get_opponent() returns
    None and the caller should skip League game (e.g. play vs frozen only).
    """
    def __init__(self, config, league_file="league.json"):
        self.config = config
        self.league_file = os.path.join(config.checkpoint_dir, league_file)
        self.opponents: List[LeagueOpponent] = []
        self.current_elo = 1200.0  # Initial Elo for the current active agent
        self.history_elo = []      # Track Elo over time [(step, elo)]
        
        self.load()

    def load(self):
        if os.path.exists(self.league_file):
            try:
                with open(self.league_file, 'r') as f:
                    data = json.load(f)
                self.current_elo = data.get('current_elo', 1200.0)
                self.history_elo = data.get('history_elo', [])
                self.opponents = []
                bad = 0
                for i, op in enumerate(data.get('opponents', [])):
                    try:
                        self.opponents.append(LeagueOpponent(**op))
                    except Exception as e:
                        bad += 1
                        if bad <= 3:
                            _log.warning("Skip bad opponent entry idx=%s: %s", i, e)
                if bad > 3:
                    _log.warning("Skipped %d bad opponent entries (file=%s)", bad, self.league_file)
                _log.info("Loaded %d opponents. Current Elo: %.1f", len(self.opponents), self.current_elo)
            except Exception as e:
                _log.error("Failed to load league file %s: %s: %s", self.league_file, type(e).__name__, e)

    def save(self):
        data = {
            'current_elo': self.current_elo,
            'history_elo': self.history_elo,
            'opponents': [vars(op) for op in self.opponents]
        }
        try:
            with open(self.league_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            _log.error("Failed to save league file %s: %s", self.league_file, e)
            try:
                with open(self.league_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e2:
                _log.error("Retry save league file failed %s: %s", self.league_file, e2)

    def update_elo(self, rating_a: float, rating_b: float, score_a: float, k_factor: float = 32.0) -> Tuple[float, float]:
        """
        Update Elo ratings for player A vs player B.
        score_a: 1.0 for win, 0.5 for draw, 0.0 for loss. Clamped to [0, 1].
        Returns new ratings (rating_a, rating_b).
        """
        score_a = max(0.0, min(1.0, float(score_a)))
        expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
        new_a = rating_a + k_factor * (score_a - expected_a)
        new_b = rating_b + k_factor * ((1.0 - score_a) - (1.0 - expected_a))
        return new_a, new_b

    def add_opponent(self, checkpoint_path: str, step: int):
        """Adds the current agent state as a new opponent in the league."""
        # Only add if we don't have too many (or replace oldest/lowest Elo?)
        # For now, keep recent N + some diverse elites.
        
        # Check if already exists
        for op in self.opponents:
            if op.step == step:
                return

        new_op = LeagueOpponent(
            checkpoint_path=checkpoint_path,
            elo=self.current_elo, # Start with current estimated Elo
            step=step,
            added_at=0.0 # TODO: use time.time()
        )
        self.opponents.append(new_op)
        
        # Max snapshots constraint
        max_snapshots = getattr(self.config, 'league_max_snapshots', 20)
        if len(self.opponents) > max_snapshots:
            # Simple eviction: remove oldest, but maybe keep highest Elo?
            # Sort by step
            self.opponents.sort(key=lambda x: x.step)
            # Remove the oldest that isn't the absolute best?
            # For simplicity, just remove oldest for now to keep moving window + elites logic later
            self.opponents.pop(0) 
            
        self.save()
        _log.info("Added opponent from step %s (Elo %.0f)", step, self.current_elo)

    def get_opponent(self) -> Optional[LeagueOpponent]:
        """Select an opponent for the current agent to play against.
        Returns None when the pool is empty; caller should skip League game."""
        if not self.opponents:
            return None
        
        # Strategy: 
        # 1. Mostly select opponents with similar Elo to encourage fair matches (learning zone).
        # 2. Occasionally select much stronger/weaker to verify robustness.
        
        # Probability weighted by Elo difference? Gaussian around current Elo?
        # Simple approach: uniform random from pool for now, as pool is small.
        # Improved: Weighted softmax based on -|delta_elo|
        
        elos = np.array([op.elo for op in self.opponents])
        diffs = np.abs(elos - self.current_elo)
        # Weights = exp(-diff / temperature)
        weights = np.exp(-diffs / 200.0) 
        weights /= weights.sum()
        
        chosen_idx = np.random.choice(len(self.opponents), p=weights)
        opp = self.opponents[chosen_idx]
        _log.info("League opponent selected: step=%s elo=%.0f (current_elo=%.0f)", opp.step, opp.elo, self.current_elo)
        return opp

    def record_match(self, opponent: LeagueOpponent, result: float):
        """
        Record result of Current Agent vs Opponent.
        result: 1.0 (Win), 0.5 (Draw), 0.0 (Loss)
        """
        # Update current agent Elo
        new_curr, new_opp = self.update_elo(self.current_elo, opponent.elo, result)
        
        diff = new_curr - self.current_elo
        self.current_elo = new_curr
        
        # Update opponent Elo in the pool
        # Find the opponent object instance and update it
        # (Since we passed the object, we can update it directly, BUT self.opponents has the references)
        opponent.elo = new_opp
        
        # Log history
        # We might call this multiple times per iteration (batch of games). 
        # Maybe average updates? For now, instantaneous updates are chaotic but work over time.
        
        # Only save periodically to avoid thrashing disk?
        # Caller handles save interval.
