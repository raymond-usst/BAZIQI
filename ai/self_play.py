"""Self-play game generation for MuZero training.

Maintainability: see ai/MAINTENANCE.md."""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Union, Any
from .game_env import EightInARowEnv
from .mcts import gumbel_muzero_search, select_action
from .replay_buffer import GameHistory
from .log_utils import get_logger

_log = get_logger(__name__)


@dataclass
class SessionContext:
    """Tracks cumulative scores across a multi-game session."""
    scores: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    game_idx: int = 0
    session_length: int = 1

    def make_context_vector(self, current_pid: int) -> np.ndarray:
        """Create a 4-dim normalized context vector for the current player.

        Returns [my_score_norm, opp1_score_norm, opp2_score_norm, games_remaining_norm].
        All values in [0, 1].
        """
        max_possible = max(1, self.session_length * 5)  # max score = winning every game
        pids = sorted(self.scores.keys())
        others = [p for p in pids if p != current_pid]

        my_score = self.scores.get(current_pid, 0) / max_possible
        opp1_score = self.scores.get(others[0], 0) / max_possible if len(others) > 0 else 0.0
        opp2_score = self.scores.get(others[1], 0) / max_possible if len(others) > 1 else 0.0
        games_remaining = (self.session_length - self.game_idx - 1) / max(1, self.session_length)

        return np.array([my_score, opp1_score, opp2_score, games_remaining], dtype=np.float32)


def play_game(network: Union[torch.nn.Module, Dict[int, torch.nn.Module]], config,
              temperature: float = 1.0,
              broadcast_fn: Optional[Callable] = None,
              game_index: int = 0,
              iteration: int = 0,
              session_context: Optional[SessionContext] = None,
              training_step: int = 0,
              board_size_override: Optional[int] = None,
              win_length_override: Optional[int] = None) -> GameHistory:
    """
    Play one complete game using MCTS.
    network: either a single MuZeroNetwork (shared) or a dict {pid: net} (KOTH mode).
    broadcast_fn(event_type, data): optional callback for live visualization.
    Returns a GameHistory with the full trajectory.

    Config (or config-like object) must have: board_size, win_length, num_players,
    local_view_size, temperature_drop_step, num_simulations, max_game_steps;
    optional: num_simulations_early/mid/late, policy_target_temp_start/end/steps.
    See ai/REUSABILITY.md.
    """
    board_size = board_size_override if board_size_override is not None else config.board_size
    win_length = win_length_override if win_length_override is not None else config.win_length

    env = EightInARowEnv(board_size=board_size, win_length=win_length)
    env.reset()
    history = GameHistory()
    history.board_size = board_size
    history.win_length = win_length

    if isinstance(network, dict):
        device = next(network[1].parameters()).device
        for net in network.values():
            net.eval()
    else:
        network.eval()
        device = next(network.parameters()).device  # cache once, reuse every move

    # Level 4: Asymmetric exploration — one random player explores more aggressively
    explorer_pid = np.random.randint(1, config.num_players + 1)  # random player 1-3
    explorer_noise_scale = 2.0   # 2x Gumbel noise for explorer
    explorer_epsilon = 0.1       # 10% random actions for explorer

    # Broadcast game start
    if broadcast_fn:
        broadcast_fn('selfplay_start', {
            'game_index': game_index,
            'iteration': iteration,
        })

    # Efficiency: cache config lookups once per game to avoid getattr in the step loop.
    temperature_drop_step = getattr(config, 'temperature_drop_step', 0)
    sims_early = getattr(config, 'num_simulations_early', config.num_simulations)
    sims_mid = getattr(config, 'num_simulations_mid', config.num_simulations)
    sims_late = getattr(config, 'num_simulations_late', config.num_simulations)
    policy_target_temp_start = getattr(config, 'policy_target_temp_start', 1.0)
    policy_target_temp_end = getattr(config, 'policy_target_temp_end', 1.0)
    policy_target_temp_steps = getattr(config, 'policy_target_temp_steps', 100000)
    max_game_steps = getattr(config, 'max_game_steps', 5000)

    step = 0
    step_retries = 0
    max_step_retries = 2
    while not env.done:
        # Determine temperature
        if step < temperature_drop_step:
            temp = temperature
        else:
            temp = 0.1  # near-greedy after warmup

        # 1. Predict View Center using Focus Network
        # Use cached rotated planes (shared between get_global_state and get_observation)
        rotated = env._get_rotated_planes_cached()
        global_state = rotated[np.newaxis, ...]  # (1, 4, 100, 100)
        gs_tensor = torch.from_numpy(global_state).to(device)
        
        if isinstance(network, dict):
            current_net = network[env.current_player_id]
        else:
            current_net = network

        try:
            cr, cc = current_net.predict_center(gs_tensor)
        except (AttributeError, ValueError) as e:
             # Fallback if network doesn't support it yet OR produces NaN
            if isinstance(e, ValueError) and 'NaN' in str(e):
                _log.error("CRITICAL: NaN detected in predict_center! Saving crash dump...")
                import os
                debug_path = f"debug_crash_actor_{os.getpid()}.pt"
                torch.save({
                    'global_state': gs_tensor.cpu(),
                    'network_state': network.state_dict(),
                    'focus_net_buffers': dict(network.focus_net.named_buffers())
                }, debug_path)
                _log.info("Saved crash dump to %s", debug_path)
                raise RuntimeError("ModelCorrupted")
            
            # Fallback to smart center
            cr, cc = env.get_smart_center(config.local_view_size)

        # 2. Validation / Fallback + get legal mask in one pass
        legal_check, legal_mask = env.get_legal_moves_and_mask(cr, cc, config.local_view_size)
        if len(legal_check) == 0:
            cr, cc = env.get_smart_center(config.local_view_size)
            legal_check, legal_mask = env.get_legal_moves_and_mask(cr, cc, config.local_view_size)

        # 3. Get Observation (pass pre-computed rotated planes to avoid recomputation)
        obs, center = env.get_observation(config.local_view_size, center=(cr, cc),
                                           rotated_planes=rotated)
        # Re-derive mask for the clamped center if it differs
        if center != (cr, cc):
            _, legal_mask = env.get_legal_moves_and_mask(center[0], center[1], config.local_view_size)

        # 3b. Compute session context vector for current player (Phase 4)
        ctx_vec = None
        if session_context is not None:
            ctx_vec = session_context.make_context_vector(env.current_player_id)

        # Dynamic simulation budget based on game progress
        if step < 10:
            dynamic_sims = sims_early
        elif step < 40:
            dynamic_sims = sims_mid
        else:
            dynamic_sims = sims_late

        # Asymmetric noise: explorer gets boosted Gumbel noise
        current_pid = env.current_player_id
        ns = explorer_noise_scale if current_pid == explorer_pid else 1.0

        # Single-step try/except: MCTS + env.step; on failure log, retry once, then return partial history
        try:
            # Run MCTS - using Gumbel MuZero (also returns root value from initial inference)
            action_probs, root_value = gumbel_muzero_search(
                current_net, obs, legal_mask, config,
                add_noise=True,
                noise_scale=ns,
                session_context_vec=ctx_vec,
                device=device,
                num_simulations_override=dynamic_sims
            )

            # Policy target temperature annealing
            # Soft targets early (high temp) → sharp targets late (low temp)
            t_start = policy_target_temp_start
            t_end = policy_target_temp_end
            t_steps = policy_target_temp_steps
            if t_start != t_end and t_steps > 0:
                progress = min(1.0, training_step / t_steps)
                policy_temp = t_start + (t_end - t_start) * progress
                target_probs = action_probs ** (1.0 / policy_temp)
                t_sum = target_probs.sum()
                if t_sum > 0:
                    target_probs /= t_sum
                else:
                    target_probs = action_probs
            else:
                target_probs = action_probs

            # Select action (from original MCTS probs, not annealed targets)
            action = select_action(action_probs, temperature=temp)

            # Explorer epsilon-greedy: 10% random legal action for diversity
            player_id = env.current_player_id
            if player_id == explorer_pid and np.random.rand() < explorer_epsilon:
                legal_indices = np.where(legal_mask > 0)[0]
                if len(legal_indices) > 0:
                    action = int(np.random.choice(legal_indices))

            # Compute threats for current state (before move)
            threats = env.get_threat_levels()

            # Store in history (with annealed policy targets)
            history.store(obs, action, 0.0, target_probs, root_value, threats, player_id, center)

            # Execute action
            board_r, board_c = env.action_to_board(action, center[0], center[1], config.local_view_size)

            # Validate the action is legal on the actual board
            if board_r < 0 or board_r >= env.BOARD_SIZE or board_c < 0 or board_c >= env.BOARD_SIZE:
                legal = env.legal_moves_local(center[0], center[1], config.local_view_size)
                if len(legal) == 0:
                    break
                board_r, board_c = legal[np.random.randint(len(legal))]
            elif env.board[board_r, board_c] != 0:
                legal = env.legal_moves_local(center[0], center[1], config.local_view_size)
                if len(legal) == 0:
                    break
                board_r, board_c = legal[np.random.randint(len(legal))]

            # Broadcast the move
            if broadcast_fn:
                broadcast_fn('selfplay_move', {
                    'row': int(board_r),
                    'col': int(board_c),
                    'player': int(player_id),
                    'step': step,
                })

            reward, done = env.step(board_r, board_c)

            # Save board snapshot periodically for fast replay buffer reconstruction
            if GameHistory.SNAPSHOT_INTERVAL > 0 and step % GameHistory.SNAPSHOT_INTERVAL == 0:
                history.board_snapshots[step] = env.board.copy()

            # Update reward for the move that just happened
            history.rewards[-1] = reward
            step += 1
            step_retries = 0
        except Exception as e:
            step_retries += 1
            _log.warning(
                "Step failed (retry %d/%d): %s: %s",
                step_retries, max_step_retries, type(e).__name__, e
            )
            if step_retries >= max_step_retries:
                _log.warning("Consecutive step failures; returning partial history.")
                break

        # Safety: prevent absurdly long games (configurable via config.max_game_steps; see SCALABILITY.md)
        if step > max_game_steps:
            break

    history.done = env.done
    history.winner = env.winner
    history.rankings = list(env.rankings)
    history.placement_rewards = dict(env.placement_rewards)
    history.final_board = env.board.copy()

    # Store session-level context for Phase 4 network awareness
    if session_context is not None:
        history.session_scores = dict(session_context.scores)
        history.session_game_idx = session_context.game_idx
        history.session_length = session_context.session_length

    # ── Terminal reward shaping using placement-based rewards ──
    # 1st: +1.0, 2nd: -0.2, 3rd: -1.0  (from 5:2:0 point mapping)
    # Assign decaying rewards to each player's last moves before game end.
    if env.winner is not None and env.placement_rewards:
        # Track which players have already received their terminal shaping
        shaped = set()
        for i in range(len(history) - 1, -1, -1):
            pid = history.player_ids[i]
            if pid not in shaped:
                base_reward = env.placement_rewards.get(pid, -1.0)
                distance = len(history) - 1 - i  # moves before end
                decay = 0.8 ** distance  # strongest at last move, decaying backwards
                history.rewards[i] = base_reward * decay
                shaped.add(pid)
            if len(shaped) >= env.NUM_PLAYERS:
                break  # All players shaped

    # Broadcast game end (with ranking & session info for dashboard)
    if broadcast_fn:
        evt = {
            'game_index': game_index,
            'winner': int(history.winner) if history.winner else None,
            'length': step,
            'rankings': [(int(pid), int(pl)) for pid, pl in env.rankings] if env.rankings else [],
        }
        if session_context is not None:
            evt['session'] = {
                'scores': {int(k): int(v) for k, v in session_context.scores.items()},
                'game_idx': session_context.game_idx,
                'session_length': session_context.session_length,
            }
        broadcast_fn('selfplay_end', evt)
        
    return history


def play_session(network: torch.nn.Module, config,
                 session_length: int,
                 temperature: float = 1.0,
                 broadcast_fn: Optional[Callable] = None,
                 game_index_base: int = 0,
                 iteration: int = 0,
                 training_step: int = 0,
                 board_size_override: Optional[int] = None,
                 win_length_override: Optional[int] = None) -> List[GameHistory]:
    """Play a session of N games, tracking cumulative scores across games.

    Each game is played with awareness of the session context (cumulative scores,
    games remaining). The resulting GameHistory objects carry session metadata
    for Phase 4 network context encoding.

    Args:
        session_length: Number of games in this session.
        game_index_base: Base index for game numbering (for logging).

    Returns:
        List of GameHistory objects, one per game in the session.
    """
    scores: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    histories: List[GameHistory] = []

    for game_idx in range(session_length):
        context = SessionContext(
            scores=dict(scores),  # snapshot current scores
            game_idx=game_idx,
            session_length=session_length,
        )

        game = play_game(
            network, config,
            temperature=temperature,
            broadcast_fn=broadcast_fn,
            game_index=game_index_base + game_idx,
            iteration=iteration,
            session_context=context,
            training_step=training_step,
            board_size_override=board_size_override,
            win_length_override=win_length_override,
        )

        # Update cumulative scores based on game rankings
        for pid, placement in game.rankings:
            scores[pid] += config.placement_points[placement]

        histories.append(game)

    return histories


def get_adaptive_session_length(training_step: int, config) -> int:
    """Compute adaptive session length based on training progress.

    Early training (0-20k steps): shorter sessions for fast iteration.
    Late training (60k+ steps): longer sessions for deeper strategy learning.
    """
    min_len = config.session_length_min
    max_len = config.session_length_max
    length = min(max_len, min_len + training_step // 20000)
    return max(min_len, length)


def run_selfplay(network: torch.nn.Module, config, num_games: int, 
                 broadcast_fn: Optional[Callable] = None,
                 iteration: int = 0) -> List[GameHistory]:
    """Run multiple self-play games."""
    # For simplicity, run sequentially. Parallelization is harder with single GPU network.
    games = []
    for i in range(num_games):
        games.append(play_game(network, config, 
                               broadcast_fn=broadcast_fn, 
                               game_index=i+1,
                               iteration=iteration))
    return games
