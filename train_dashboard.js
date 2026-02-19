/**
 * Training Dashboard - Real-time visualization
 * Connects via WebSocket to MuZero training process
 *
 * Supports session-based scoring (5:2:0 placement system)
 * and per-player cumulative score tracking.
 */
(function () {
    'use strict';

    // ---- Config ----
    const BOARD_SIZE = 100;
    const PLAYER_COLORS = { 1: '#e74c3c', 2: '#27ae60', 3: '#2980b9' };
    const PLAYER_NAMES = { 1: '红方', 2: '绿方', 3: '蓝方' };
    const PLAYER_GLOWS = { 1: '#ff6b6b', 2: '#6bff6b', 3: '#6bb5ff' };
    const PLACEMENT_POINTS = [5, 2, 0]; // 1st, 2nd, 3rd

    // ---- ELO Config ----
    const ELO_INITIAL = 1500;
    const ELO_K = 16;  // K-factor per pairwise matchup (each player faces 2 opponents per game)

    // ---- State ----
    let ws = null;
    let board = [];          // 100x100
    let boardCanvas, boardCtx;
    let gameStep = 0;
    let currentPlayer = 1;
    let totalGames = 0;
    let winCounts = { 1: 0, 2: 0, 3: 0, draw: 0 };
    let lossChart, statsChart, scoresChart, eloChart;
    let startTime = Date.now();
    let lastMove = null;

    // Session / Placement tracking
    let placementCounts = {
        1: [0, 0, 0], // [1st, 2nd, 3rd] for player 1
        2: [0, 0, 0],
        3: [0, 0, 0],
    };
    let cumulativePoints = { 1: 0, 2: 0, 3: 0 };
    let rankedGames = 0; // games with ranking data
    let currentSessionInfo = null; // latest session context

    // ELO tracking
    let eloRatings = { 1: ELO_INITIAL, 2: ELO_INITIAL, 3: ELO_INITIAL };
    let lastEloRankedGames = 0; // prevent re-computation with same data
    const eloHistory = { labels: [], red: [], green: [], blue: [] };
    const MAX_ELO_POINTS = 500;

    // Rolling score history for chart (game-by-game)
    const scoreHistory = { labels: [], red: [], green: [], blue: [] };
    const MAX_SCORE_POINTS = 300;

    // ---- Init ----
    function init() {
        boardCanvas = document.getElementById('live-board');
        boardCtx = boardCanvas.getContext('2d');
        resetBoard();
        initCharts();
        setupEvents();
        renderBoard();
    }

    function resetBoard() {
        board = Array.from({ length: BOARD_SIZE }, () => new Array(BOARD_SIZE).fill(0));
        gameStep = 0;
        lastMove = null;
    }

    // ---- WebSocket ----
    function setupEvents() {
        document.getElementById('btn-connect').addEventListener('click', connect);
        // Auto-connect on load
        setTimeout(connect, 500);
    }

    function connect() {
        const url = document.getElementById('ws-url').value;
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            ws.close();
        }

        setStatus('connecting');
        try {
            ws = new WebSocket(url);
        } catch (e) {
            setStatus('disconnected');
            return;
        }

        ws.onopen = () => setStatus('connected');
        ws.onclose = () => {
            setStatus('disconnected');
            // Auto-reconnect after 3s
            setTimeout(() => {
                if (!ws || ws.readyState === WebSocket.CLOSED) connect();
            }, 3000);
        };
        ws.onerror = () => setStatus('disconnected');
        ws.onmessage = (evt) => {
            try {
                const msg = JSON.parse(evt.data);
                handleMessage(msg);
            } catch (e) {
                console.warn('Bad WS message:', e);
            }
        };
    }

    function setStatus(status) {
        const dot = document.getElementById('ws-status');
        const label = document.getElementById('ws-label');
        dot.className = 'status-dot ' + status;
        const labels = { connected: '已连接', disconnected: '未连接', connecting: '连接中...' };
        label.textContent = labels[status] || status;
    }

    // ---- Message Handler ----
    function handleMessage(msg) {
        switch (msg.type) {
            case 'selfplay_start':
                resetBoard();
                document.getElementById('board-info').textContent = `Game #${msg.game_index + 1}`;
                document.getElementById('game-iter').textContent = `Iteration: ${msg.iteration || '-'}`;
                renderBoard();
                break;

            case 'selfplay_move':
                placeStone(msg.row, msg.col, msg.player);
                lastMove = { row: msg.row, col: msg.col };
                gameStep++;
                currentPlayer = msg.player;
                document.getElementById('game-step').textContent = `步数: ${gameStep}`;
                document.getElementById('game-player').textContent = `落子: ${PLAYER_NAMES[msg.player] || '-'}`;
                renderBoard();
                break;

            case 'selfplay_end':
                totalGames++;
                if (msg.winner && msg.winner >= 1 && msg.winner <= 3) {
                    winCounts[msg.winner]++;
                } else {
                    winCounts.draw++;
                }
                updateWinRates();

                // Process rankings & session data
                if (msg.rankings && msg.rankings.length > 0) {
                    processRankings(msg.rankings);
                }
                if (msg.session) {
                    currentSessionInfo = msg.session;
                    updateSessionDisplay();
                }

                document.getElementById('board-info').textContent =
                    msg.winner ? `${PLAYER_NAMES[msg.winner]} 获胜 (${gameStep}步)` : `平局 (${gameStep}步)`;
                break;

            case 'training_metrics':
                updateLossChart(msg);
                updateStatsChart(msg);
                updateSummaryBar(msg);
                // Sync authoritative placement data from all actors (not just Actor 0)
                if (msg.placements && msg.ranked_games) {
                    syncPlacementsFromServer(msg.ranked_games, msg.placements);
                    updateScoresChart();
                    // Replay ELO from authoritative server placements
                    replayEloFromPlacements(msg.placements, msg.ranked_games);
                    updateEloChart();
                    updateEloBadge();
                }
                break;

            case 'batch_stats':
                // Async mode: stats from all actors
                if (msg.total_games) totalGames = msg.total_games;
                if (msg.win_counts) winCounts = { ...winCounts, ...msg.win_counts };
                updateWinRates();
                // Sync placement/ELO from all actors
                if (msg.placements && msg.ranked_games) {
                    syncPlacementsFromServer(msg.ranked_games, msg.placements);
                    updateScoresChart();
                    replayEloFromPlacements(msg.placements, msg.ranked_games);
                    updateEloChart();
                    updateEloBadge();
                }
                break;

            case 'status':
                // Initial status update
                if (msg.total_games) totalGames = msg.total_games;
                if (msg.win_counts) winCounts = { ...winCounts, ...msg.win_counts };
                updateWinRates();
                updateSummaryBar(msg);
                // Sync placement/ELO from all actors
                if (msg.placements && msg.ranked_games) {
                    syncPlacementsFromServer(msg.ranked_games, msg.placements);
                    updateScoresChart();
                    replayEloFromPlacements(msg.placements, msg.ranked_games);
                    updateEloChart();
                    updateEloBadge();
                }
                break;

            case 'metrics_history':
                // Bulk load history
                if (msg.data && msg.data.length > 0) {
                    console.log(`Loaded ${msg.data.length} metrics history entries.`);

                    // 1. Populate loss & stats charts
                    msg.data.forEach(m => {
                        updateLossChart(m);
                        updateStatsChart(m);
                    });

                    // 2. Refresh charts once
                    lossChart.update();
                    statsChart.update();

                    // 3. Restore win stats from last entry
                    const last = msg.data[msg.data.length - 1];
                    if (last.total_games) totalGames = last.total_games;
                    if (last.win_counts) winCounts = { ...winCounts, ...last.win_counts };
                    updateWinRates();
                    updateSummaryBar(last);

                    // 4. Restore placements, scores chart & ELO from history
                    //    Find entries with placement data and replay them
                    //    Reset ELO to initial before replaying
                    eloRatings = { 1: ELO_INITIAL, 2: ELO_INITIAL, 3: ELO_INITIAL };
                    lastEloRankedGames = 0;
                    eloHistory.labels.length = 0;
                    eloHistory.red.length = 0;
                    eloHistory.green.length = 0;
                    eloHistory.blue.length = 0;
                    scoreHistory.labels.length = 0;
                    scoreHistory.red.length = 0;
                    scoreHistory.green.length = 0;
                    scoreHistory.blue.length = 0;

                    let lastRankedGames = 0;
                    for (const entry of msg.data) {
                        if (entry.placements && entry.ranked_games) {
                            // Sync placement counts
                            rankedGames = entry.ranked_games;
                            for (let pid = 1; pid <= 3; pid++) {
                                const counts = entry.placements[String(pid)] || [0, 0, 0];
                                placementCounts[pid] = [...counts];
                                cumulativePoints[pid] = counts[0] * PLACEMENT_POINTS[0]
                                    + counts[1] * PLACEMENT_POINTS[1]
                                    + counts[2] * PLACEMENT_POINTS[2];
                            }

                            // Replay ELO for new games since last entry
                            const newGames = entry.ranked_games - lastRankedGames;
                            if (newGames > 0) {
                                // Estimate per-game ELO from placement proportions
                                replayEloFromPlacements(entry.placements, entry.ranked_games);
                            }
                            lastRankedGames = entry.ranked_games;

                            // Push to score history
                            scoreHistory.labels.push(entry.ranked_games);
                            scoreHistory.red.push(cumulativePoints[1]);
                            scoreHistory.green.push(cumulativePoints[2]);
                            scoreHistory.blue.push(cumulativePoints[3]);
                            if (scoreHistory.labels.length > MAX_SCORE_POINTS) {
                                scoreHistory.labels.shift();
                                scoreHistory.red.shift();
                                scoreHistory.green.shift();
                                scoreHistory.blue.shift();
                            }
                        }
                    }

                    // Update all displays
                    updatePlacementDisplay();
                    if (scoresChart) {
                        scoresChart.data.labels = [...scoreHistory.labels];
                        scoresChart.data.datasets[0].data = [...scoreHistory.red];
                        scoresChart.data.datasets[1].data = [...scoreHistory.green];
                        scoresChart.data.datasets[2].data = [...scoreHistory.blue];
                        scoresChart.update();
                    }
                    updateEloChart();
                    updateEloBadge();
                }
                break;
        }
    }

    // ---- Rankings & Session Processing ----
    function processRankings(rankings) {
        // rankings: [[pid, placement], ...] where placement is 0-indexed (0=1st, 1=2nd, 2=3rd)
        rankedGames++;

        for (const [pid, placement] of rankings) {
            if (pid >= 1 && pid <= 3 && placement >= 0 && placement <= 2) {
                placementCounts[pid][placement]++;
                cumulativePoints[pid] += PLACEMENT_POINTS[placement];
            }
        }

        // Update placement display only (charts updated by server data via batch_stats/status/training_metrics)
        updatePlacementDisplay();
    }

    function syncPlacementsFromServer(serverRankedGames, serverPlacements) {
        /**
         * Overwrite local placement/score state with authoritative data from all actors.
         * This ensures the placement panel matches the win-rate panel (both cover all actors).
         * serverPlacements: {'1': [1st, 2nd, 3rd], '2': [...], '3': [...]}
         */
        rankedGames = serverRankedGames;
        for (let pid = 1; pid <= 3; pid++) {
            const counts = serverPlacements[String(pid)] || [0, 0, 0];
            placementCounts[pid] = [...counts];
            cumulativePoints[pid] = counts[0] * PLACEMENT_POINTS[0]
                + counts[1] * PLACEMENT_POINTS[1]
                + counts[2] * PLACEMENT_POINTS[2];
        }
        updatePlacementDisplay();
    }

    // ---- ELO Rating System ----
    // 3-player ELO: decompose into pairwise matchups.
    // For placements [1st, 2nd, 3rd], we have 3 pairs:
    //   1st beats 2nd, 1st beats 3rd, 2nd beats 3rd.
    // Standard ELO update applied to each pair with K-factor per pair.
    function updateElo(rankings) {
        if (rankings.length < 2) return;

        // Build placement map: pid → placement (0=1st, 1=2nd, 2=3rd)
        const plMap = {};
        for (const [pid, pl] of rankings) {
            plMap[pid] = pl;
        }
        const pids = Object.keys(plMap).map(Number);

        // Accumulate deltas for each player (apply all at once to avoid order bias)
        const deltas = {};
        for (const p of pids) deltas[p] = 0;

        // For every pair (A, B) where A placed higher (lower index) than B
        for (let i = 0; i < pids.length; i++) {
            for (let j = i + 1; j < pids.length; j++) {
                const pidA = pids[i];
                const pidB = pids[j];
                // Determine who placed higher
                let winner, loser;
                if (plMap[pidA] < plMap[pidB]) {
                    winner = pidA; loser = pidB;
                } else if (plMap[pidB] < plMap[pidA]) {
                    winner = pidB; loser = pidA;
                } else {
                    // Same placement (shouldn't happen, but treat as draw)
                    const eA = expectedScore(eloRatings[pidA], eloRatings[pidB]);
                    const eB = 1 - eA;
                    deltas[pidA] += ELO_K * (0.5 - eA);
                    deltas[pidB] += ELO_K * (0.5 - eB);
                    continue;
                }

                const eW = expectedScore(eloRatings[winner], eloRatings[loser]);
                const eL = 1 - eW;
                deltas[winner] += ELO_K * (1 - eW);  // won: score=1
                deltas[loser] += ELO_K * (0 - eL);  // lost: score=0
            }
        }

        // Apply deltas
        for (const p of pids) {
            eloRatings[p] = Math.max(100, eloRatings[p] + deltas[p]); // floor at 100
        }

        // Push to history & update chart
        eloHistory.labels.push(rankedGames);
        eloHistory.red.push(Math.round(eloRatings[1]));
        eloHistory.green.push(Math.round(eloRatings[2]));
        eloHistory.blue.push(Math.round(eloRatings[3]));

        if (eloHistory.labels.length > MAX_ELO_POINTS) {
            eloHistory.labels.shift();
            eloHistory.red.shift();
            eloHistory.green.shift();
            eloHistory.blue.shift();
        }

        updateEloChart();
        updateEloBadge();
    }

    /**
     * Derive approximate ELO ratings from cumulative placement data.
     * Since we don't have individual game results in history, we estimate
     * from aggregate placement proportions.
     */
    function replayEloFromPlacements(placements, totalRanked) {
        if (totalRanked <= 0) return;

        // Skip if we've already processed this exact ranked_games count
        if (totalRanked === lastEloRankedGames) return;
        lastEloRankedGames = totalRanked;

        // Calculate win rates per player from placements
        // Player with more 1st-place finishes should have higher ELO
        const rates = {};
        for (let pid = 1; pid <= 3; pid++) {
            const counts = placements[String(pid)] || [0, 0, 0];
            // Score: 1st=1.0, 2nd=0.5, 3rd=0.0 (analogous to chess)
            rates[pid] = (counts[0] * 1.0 + counts[1] * 0.5 + counts[2] * 0.0) / totalRanked;
        }

        // Compute pairwise performance and derive ELO differences
        const pids = [1, 2, 3];
        for (let i = 0; i < pids.length; i++) {
            for (let j = i + 1; j < pids.length; j++) {
                const pidA = pids[i];
                const pidB = pids[j];
                // Estimate pairwise result from placement rates
                const pA = rates[pidA];
                const pB = rates[pidB];
                const totalP = pA + pB;
                if (totalP <= 0) continue;
                const scoreA = pA / totalP; // A's pairwise win rate vs B
                const eA = expectedScore(eloRatings[pidA], eloRatings[pidB]);
                const eB = 1 - eA;
                const scoreB = 1 - scoreA;
                eloRatings[pidA] += ELO_K * (scoreA - eA);
                eloRatings[pidB] += ELO_K * (scoreB - eB);
            }
        }

        // Floor at 100
        for (const p of pids) {
            eloRatings[p] = Math.max(100, eloRatings[p]);
        }

        // Push to history (dedup: update last point if same X-value)
        const lastLabel = eloHistory.labels.length > 0 ? eloHistory.labels[eloHistory.labels.length - 1] : -1;
        if (lastLabel === totalRanked) {
            // Same X-value: update in-place
            eloHistory.red[eloHistory.red.length - 1] = Math.round(eloRatings[1]);
            eloHistory.green[eloHistory.green.length - 1] = Math.round(eloRatings[2]);
            eloHistory.blue[eloHistory.blue.length - 1] = Math.round(eloRatings[3]);
        } else {
            // New X-value: push new point
            eloHistory.labels.push(totalRanked);
            eloHistory.red.push(Math.round(eloRatings[1]));
            eloHistory.green.push(Math.round(eloRatings[2]));
            eloHistory.blue.push(Math.round(eloRatings[3]));

            if (eloHistory.labels.length > MAX_ELO_POINTS) {
                eloHistory.labels.shift();
                eloHistory.red.shift();
                eloHistory.green.shift();
                eloHistory.blue.shift();
            }
        }
    }

    function expectedScore(ratingA, ratingB) {
        return 1.0 / (1.0 + Math.pow(10, (ratingB - ratingA) / 400));
    }

    function updateEloChart() {
        if (!eloChart) return;
        eloChart.data.labels = [...eloHistory.labels];
        eloChart.data.datasets[0].data = [...eloHistory.red];
        eloChart.data.datasets[1].data = [...eloHistory.green];
        eloChart.data.datasets[2].data = [...eloHistory.blue];
        eloChart.update('none');
    }

    function updateEloBadge() {
        const r = Math.round(eloRatings[1]);
        const g = Math.round(eloRatings[2]);
        const b = Math.round(eloRatings[3]);
        const badge = document.getElementById('elo-badge');
        if (badge) badge.textContent = `${r} / ${g} / ${b}`;
        const sumEl = document.getElementById('sum-elo');
        if (sumEl) sumEl.textContent = `ELO: ${r}/${g}/${b}`;
    }

    function updatePlacementDisplay() {
        const el = document.getElementById('placement-total');
        if (el) el.textContent = `${rankedGames} 局`;

        for (let pid = 1; pid <= 3; pid++) {
            const counts = placementCounts[pid];
            for (let pl = 0; pl < 3; pl++) {
                const cell = document.getElementById(`plc-${pid}-${pl}`);
                if (cell) cell.textContent = counts[pl];
            }
            // Average points per game
            const avg = rankedGames > 0 ? (cumulativePoints[pid] / rankedGames).toFixed(1) : '0.0';
            const avgCell = document.getElementById(`plc-${pid}-avg`);
            if (avgCell) avgCell.textContent = avg;
        }
    }

    function updateScoresChart() {
        const gameNum = rankedGames;
        const lastLabel = scoreHistory.labels.length > 0 ? scoreHistory.labels[scoreHistory.labels.length - 1] : -1;

        if (lastLabel === gameNum) {
            // Same X-value: update last point in-place (dedup)
            scoreHistory.red[scoreHistory.red.length - 1] = cumulativePoints[1];
            scoreHistory.green[scoreHistory.green.length - 1] = cumulativePoints[2];
            scoreHistory.blue[scoreHistory.blue.length - 1] = cumulativePoints[3];
        } else {
            // New X-value: push new point
            scoreHistory.labels.push(gameNum);
            scoreHistory.red.push(cumulativePoints[1]);
            scoreHistory.green.push(cumulativePoints[2]);
            scoreHistory.blue.push(cumulativePoints[3]);

            // Trim to max points
            if (scoreHistory.labels.length > MAX_SCORE_POINTS) {
                scoreHistory.labels.shift();
                scoreHistory.red.shift();
                scoreHistory.green.shift();
                scoreHistory.blue.shift();
            }
        }

        if (scoresChart) {
            scoresChart.data.labels = [...scoreHistory.labels];
            scoresChart.data.datasets[0].data = [...scoreHistory.red];
            scoresChart.data.datasets[1].data = [...scoreHistory.green];
            scoresChart.data.datasets[2].data = [...scoreHistory.blue];
            scoresChart.update('none');
        }
    }

    function updateSessionDisplay() {
        if (!currentSessionInfo) return;
        const { scores, game_idx, session_length } = currentSessionInfo;
        const infoEl = document.getElementById('session-info');
        if (infoEl) {
            infoEl.textContent = `局 ${game_idx + 1}/${session_length}`;
        }
        const sumEl = document.getElementById('sum-session');
        if (sumEl) {
            const s = scores || {};
            sumEl.textContent = `会话: ${game_idx + 1}/${session_length} [${s[1] || 0}:${s[2] || 0}:${s[3] || 0}]`;
        }
    }

    // ---- Board Rendering ----
    function placeStone(row, col, player) {
        if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
            board[row][col] = player;
        }
    }

    function renderBoard() {
        // Find bounding box of occupied cells
        let minR = BOARD_SIZE, maxR = 0, minC = BOARD_SIZE, maxC = 0;
        let hasStones = false;
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                if (board[r][c] !== 0) {
                    hasStones = true;
                    minR = Math.min(minR, r);
                    maxR = Math.max(maxR, r);
                    minC = Math.min(minC, c);
                    maxC = Math.max(maxC, c);
                }
            }
        }

        if (!hasStones) {
            // Empty board - show center region
            minR = 45; maxR = 55; minC = 45; maxC = 55;
        }

        // Add padding
        const pad = 5;
        minR = Math.max(0, minR - pad);
        maxR = Math.min(BOARD_SIZE - 1, maxR + pad);
        minC = Math.max(0, minC - pad);
        maxC = Math.min(BOARD_SIZE - 1, maxC + pad);

        const viewRows = maxR - minR + 1;
        const viewCols = maxC - minC + 1;

        // Calculate cell size to fit in canvas
        const wrapper = document.getElementById('live-board-wrapper');
        const maxW = wrapper.clientWidth - 24;
        const maxH = wrapper.clientHeight - 24;
        const cellSize = Math.max(3, Math.min(
            Math.floor(maxW / viewCols),
            Math.floor(maxH / viewRows),
            20
        ));

        boardCanvas.width = viewCols * cellSize;
        boardCanvas.height = viewRows * cellSize;

        // Background (wood texture)
        boardCtx.fillStyle = '#c19a6b';
        boardCtx.fillRect(0, 0, boardCanvas.width, boardCanvas.height);

        // Grid lines
        boardCtx.strokeStyle = 'rgba(0,0,0,0.15)';
        boardCtx.lineWidth = 0.5;
        for (let r = 0; r < viewRows; r++) {
            const y = r * cellSize + cellSize / 2;
            boardCtx.beginPath();
            boardCtx.moveTo(0, y);
            boardCtx.lineTo(viewCols * cellSize, y);
            boardCtx.stroke();
        }
        for (let c = 0; c < viewCols; c++) {
            const x = c * cellSize + cellSize / 2;
            boardCtx.beginPath();
            boardCtx.moveTo(x, 0);
            boardCtx.lineTo(x, viewRows * cellSize);
            boardCtx.stroke();
        }

        // Stones
        const stoneRadius = cellSize * 0.4;
        for (let r = minR; r <= maxR; r++) {
            for (let c = minC; c <= maxC; c++) {
                const p = board[r][c];
                if (p === 0) continue;
                const cx = (c - minC) * cellSize + cellSize / 2;
                const cy = (r - minR) * cellSize + cellSize / 2;

                // Shadow
                boardCtx.beginPath();
                boardCtx.arc(cx + 1, cy + 1, stoneRadius, 0, Math.PI * 2);
                boardCtx.fillStyle = 'rgba(0,0,0,0.25)';
                boardCtx.fill();

                // Stone
                const grad = boardCtx.createRadialGradient(
                    cx - stoneRadius * 0.3, cy - stoneRadius * 0.3, stoneRadius * 0.1,
                    cx, cy, stoneRadius
                );
                grad.addColorStop(0, PLAYER_GLOWS[p] || '#fff');
                grad.addColorStop(1, PLAYER_COLORS[p] || '#888');
                boardCtx.beginPath();
                boardCtx.arc(cx, cy, stoneRadius, 0, Math.PI * 2);
                boardCtx.fillStyle = grad;
                boardCtx.fill();
            }
        }

        // Highlight last move
        if (lastMove) {
            const { row, col } = lastMove;
            const cx = (col - minC) * cellSize + cellSize / 2;
            const cy = (row - minR) * cellSize + cellSize / 2;

            boardCtx.beginPath();
            boardCtx.arc(cx, cy, stoneRadius + 2, 0, Math.PI * 2);
            boardCtx.strokeStyle = 'rgba(255,255,255,0.9)';
            boardCtx.lineWidth = 2.5;
            boardCtx.stroke();
        }
    }

    // ---- Win Rates ----
    function updateWinRates() {
        document.getElementById('total-games').textContent = `${totalGames} 局`;
        if (totalGames === 0) return;

        const rates = {
            red: (winCounts[1] / totalGames) * 100,
            green: (winCounts[2] / totalGames) * 100,
            blue: (winCounts[3] / totalGames) * 100,
            draw: (winCounts.draw / totalGames) * 100,
        };

        for (const [key, pct] of Object.entries(rates)) {
            document.getElementById(`wr-${key}`).style.width = pct + '%';
            document.getElementById(`wr-${key}-pct`).textContent = pct.toFixed(1) + '%';
        }
    }

    // ---- Charts ----
    function initCharts() {
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#8a8aaa', font: { size: 9, family: 'JetBrains Mono' }, maxTicksLimit: 8 },
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#8a8aaa', font: { size: 9, family: 'JetBrains Mono' } },
                },
            },
            plugins: {
                legend: {
                    labels: { color: '#ccc', font: { size: 10 }, boxWidth: 10, padding: 6 },
                    position: 'top',
                },
            },
        };

        // Loss chart
        lossChart = new Chart(document.getElementById('loss-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Total', data: [], borderColor: '#a29bfe', borderWidth: 2, pointRadius: 0, tension: 0.3 },
                    { label: 'Value', data: [], borderColor: '#ff6b6b', borderWidth: 1.5, pointRadius: 0, tension: 0.3, hidden: true },
                    { label: 'Reward', data: [], borderColor: '#6bff6b', borderWidth: 1.5, pointRadius: 0, tension: 0.3, hidden: true },
                    { label: 'Policy', data: [], borderColor: '#6bb5ff', borderWidth: 1.5, pointRadius: 0, tension: 0.3, hidden: true },
                    { label: 'Focus', data: [], borderColor: '#f1c40f', borderWidth: 1.5, pointRadius: 0, tension: 0.3, hidden: false },
                ],
            },
            options: {
                ...commonOptions,
                scales: {
                    ...commonOptions.scales,
                    y: { ...commonOptions.scales.y, type: 'logarithmic' },
                },
            },
        });

        // Scores chart (cumulative points per player)
        scoresChart = new Chart(document.getElementById('scores-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '红方', data: [],
                        borderColor: '#e74c3c', backgroundColor: 'rgba(231,76,60,0.08)',
                        borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
                    },
                    {
                        label: '绿方', data: [],
                        borderColor: '#27ae60', backgroundColor: 'rgba(39,174,96,0.08)',
                        borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
                    },
                    {
                        label: '蓝方', data: [],
                        borderColor: '#2980b9', backgroundColor: 'rgba(41,128,185,0.08)',
                        borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
                    },
                ],
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        ...commonOptions.scales.x,
                        title: { display: true, text: '局数', color: '#8a8aaa', font: { size: 9 } },
                    },
                    y: {
                        ...commonOptions.scales.y,
                        title: { display: true, text: '累计积分', color: '#8a8aaa', font: { size: 9 } },
                    },
                },
            },
        });

        // ELO chart (per-player rating curves)
        eloChart = new Chart(document.getElementById('elo-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '红方', data: [],
                        borderColor: '#e74c3c', backgroundColor: 'rgba(231,76,60,0.06)',
                        borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false,
                    },
                    {
                        label: '绿方', data: [],
                        borderColor: '#27ae60', backgroundColor: 'rgba(39,174,96,0.06)',
                        borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false,
                    },
                    {
                        label: '蓝方', data: [],
                        borderColor: '#2980b9', backgroundColor: 'rgba(41,128,185,0.06)',
                        borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false,
                    },
                ],
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        ...commonOptions.scales.x,
                        title: { display: true, text: '局数', color: '#8a8aaa', font: { size: 9 } },
                    },
                    y: {
                        ...commonOptions.scales.y,
                        title: { display: true, text: 'ELO', color: '#8a8aaa', font: { size: 9 } },
                    },
                },
                plugins: {
                    ...commonOptions.plugins,
                    // Reference line at 1500
                    annotation: undefined,
                },
            },
        });

        // Stats chart (game length + LR)
        statsChart = new Chart(document.getElementById('stats-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: '游戏长度', data: [], borderColor: '#f39c12', borderWidth: 2, pointRadius: 0, tension: 0.3, yAxisID: 'y' },
                    { label: '红方胜率%', data: [], borderColor: '#e74c3c', borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: 'y1' },
                    { label: '绿方胜率%', data: [], borderColor: '#27ae60', borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: 'y1' },
                    { label: '蓝方胜率%', data: [], borderColor: '#2980b9', borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: 'y1' },
                ],
            },
            options: {
                ...commonOptions,
                scales: {
                    x: commonOptions.scales.x,
                    y: {
                        ...commonOptions.scales.y,
                        position: 'left',
                        title: { display: true, text: '游戏长度', color: '#8a8aaa', font: { size: 9 } },
                    },
                    y1: {
                        ...commonOptions.scales.y,
                        position: 'right',
                        min: 0, max: 100,
                        title: { display: true, text: '胜率%', color: '#8a8aaa', font: { size: 9 } },
                        grid: { drawOnChartArea: false },
                    },
                },
            },
        });
    }

    function updateLossChart(metrics) {
        const label = `${metrics.step || ''}`;
        lossChart.data.labels.push(label);
        lossChart.data.datasets[0].data.push(metrics.loss);
        lossChart.data.datasets[1].data.push(metrics.loss_value);
        lossChart.data.datasets[2].data.push(metrics.loss_reward);
        lossChart.data.datasets[3].data.push(metrics.loss_policy);
        lossChart.data.datasets[4].data.push(metrics.loss_focus);

        // Keep max 200 points
        if (lossChart.data.labels.length > 200) {
            lossChart.data.labels.shift();
            lossChart.data.datasets.forEach(d => d.data.shift());
        }
        lossChart.update('none');

        document.getElementById('current-loss').textContent = `Loss: ${metrics.loss.toFixed(4)} | F: ${metrics.loss_focus.toFixed(4)}`;
    }

    function updateStatsChart(metrics) {
        const label = `${metrics.step || ''}`;
        statsChart.data.labels.push(label);
        statsChart.data.datasets[0].data.push(metrics.avg_game_length || 0);

        // Cumulative win rates
        const total = totalGames || 1;
        statsChart.data.datasets[1].data.push((winCounts[1] / total) * 100);
        statsChart.data.datasets[2].data.push((winCounts[2] / total) * 100);
        statsChart.data.datasets[3].data.push((winCounts[3] / total) * 100);

        if (statsChart.data.labels.length > 200) {
            statsChart.data.labels.shift();
            statsChart.data.datasets.forEach(d => d.data.shift());
        }
        statsChart.update('none');
    }

    function updateSummaryBar(metrics) {
        if (metrics.step !== undefined)
            document.getElementById('sum-step').textContent = `Step: ${metrics.step}`;
        if (metrics.iteration !== undefined)
            document.getElementById('sum-iter').textContent = `Iteration: ${metrics.iteration}`;
        if (metrics.lr !== undefined)
            document.getElementById('sum-lr').textContent = `LR: ${metrics.lr.toFixed(6)}`;
        if (metrics.buffer_games !== undefined)
            document.getElementById('sum-buf').textContent = `Buffer: ${metrics.buffer_games}`;

        const elapsed = ((Date.now() - startTime) / 60000).toFixed(1);
        document.getElementById('sum-time').textContent = `运行时间: ${elapsed}min`;
    }

    // ---- Start ----
    window.addEventListener('DOMContentLoaded', init);
    // Resize handler
    window.addEventListener('resize', () => renderBoard());
})();
