
import sys
import os
import unittest
import torch
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.muzero_config import MuZeroConfig
from ai.pbt import Population

class TestPBTKOTHIntegration(unittest.TestCase):
    def setUp(self):
        self.config = MuZeroConfig()
        self.config.pbt_population_size = 4
        self.config.pbt_period = 10
        self.config.koth_mode = True
        self.config.koth_period = 3
        self.config.device = 'cpu'
        
        self.population = Population(self.config)
        # Initialize dummy model
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def test_scheduling_logic(self):
        """
        Simulate the Learner Loop scheduling logic.
        Verify:
        1. KOTH rotates at step 3, 6, 9.
        2. PBT evolves at step 10.
        3. PBT reset clears KOTH state (resets active_pid to 1).
        """
        
        # State variables from learner loop
        pbt_step_counter = 0
        active_agent_idx = 0
        
        koth_step_counter = 0
        koth_active_pid = 1
        frozen_models = {}
        
        # Mock initial frozen models (normally done at start)
        for pid in [1, 2, 3]:
            frozen_models[pid] = {k: v.clone() for k, v in self.model.state_dict().items()}

        history = []
        
        for step in range(1, 21):
            event = []
            
            # --- PBT Logic ---
            pbt_step_counter += 1
            if pbt_step_counter >= self.config.pbt_period:
                # Evolve
                self.population.sync_agent_weights(active_agent_idx, self.model, self.optimizer)
                self.population.exploit_and_explore()
                
                # Rotate
                active_agent_idx = (active_agent_idx + 1) % self.population.size
                event.append(f"PBT_EVOLVE->{active_agent_idx}")
                
                # Reset PBT
                pbt_step_counter = 0
                
                # Re-init KOTH
                koth_active_pid = 1
                koth_step_counter = 0
                # Refresh frozen from population
                for pid in [1, 2, 3]:
                    a_idx = (pid - 1) % self.population.size
                    frozen_models[pid] = copy.deepcopy(self.population.agents[a_idx].model_state)
                event.append("KOTH_RESET")

            # --- KOTH Logic ---
            koth_step_counter += 1
            if koth_step_counter >= self.config.koth_period:
                old = koth_active_pid
                koth_active_pid = (koth_active_pid % 3) + 1
                koth_step_counter = 0
                event.append(f"KOTH_ROTATE {old}->{koth_active_pid}")
                
                # Snapshot
                frozen_models[old] = copy.deepcopy(self.model.state_dict())

            history.append((step, koth_active_pid, active_agent_idx, event))

        # Verification
        
        # Step 3: KOTH Rotate 1->2
        self.assertIn("KOTH_ROTATE 1->2", history[2][3])
        self.assertEqual(history[2][1], 2) # active pid
        
        # Step 6: KOTH Rotate 2->3
        self.assertIn("KOTH_ROTATE 2->3", history[5][3])
        self.assertEqual(history[5][1], 3)
        
        # Step 9: KOTH Rotate 3->1
        self.assertIn("KOTH_ROTATE 3->1", history[8][3])
        self.assertEqual(history[8][1], 1)
        
        # Step 10: PBT Evolve
        # KOTH step was 1 (reset at step 9). At step 10, koth_step becomes 1. PBT triggers.
        # Wait, PBT logic is BEFORE KOTH logic in train_async.py?
        # Yes, lines 797 (PBT) then 867 (KOTH).
        # At step 10:
        # PBT counter becomes 10 -> Triggers.
        #   - Rotates active agent 0->1.
        #   - Resets KOTH active_pid -> 1.
        #   - Resets KOTH counter -> 0.
        # Then KOTH logic:
        #   - Increments counter 0->1.
        #   - Check 1 >= 3? No.
        # So at end of Step 10: active_pid should be 1. active_agent should be 1.
        
        self.assertIn(f"PBT_EVOLVE->1", history[9][3])
        self.assertIn("KOTH_RESET", history[9][3])
        self.assertEqual(history[9][1], 1) # koth active pid
        self.assertEqual(history[9][2], 1) # pbt active agent
        
        # Step 13: KOTH Rotate 1->2 (Counter: 10->1, 11->2, 12->3 => Rotate)
        # wait.
        # Step 10: PBT resets KOTH counter to 0. Then KOTH logic increments to 1.
        # Step 11: KOTH counter 1->2.
        # Step 12: KOTH counter 2->3. Rotate!
        # So Step 12 (index 11) should have KOTH_ROTATE.
        
        self.assertIn("KOTH_ROTATE 1->2", history[11][3])
        
if __name__ == '__main__':
    unittest.main()
