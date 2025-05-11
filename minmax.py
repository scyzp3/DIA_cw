# -*- coding: utf-8 -*-
"""
Optimized Minimax algorithm for Gomoku
Enhanced defensive strategy with improved two-stone patterns
"""

import numpy as np
import time
import copy


class MinimaxPlayer:
    """Minimax player with enhanced defensive logic"""

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.player = None
        self.nodes_evaluated = 0
        self.time_spent = 0

    def set_player_ind(self, p):
        """Set player index"""
        self.player = p

    def reset_player(self):
        """Reset player state"""
        pass

    def get_action(self, board):
        """Get next move"""
        self.nodes_evaluated = 0
        start_time = time.time()

        # Get legal moves
        moves = board.availables
        if not moves:
            return -1

        # First move - pick center
        if len(moves) == board.width * board.height:
            return board.height // 2 * board.width + board.width // 2

        # Check for critical moves (immediate threats)
        critical = self._check_critical_move(board)
        if critical is not None:
            self.time_spent = time.time() - start_time
            return critical

        # Find best move using minimax
        best_score = -float('inf')
        best_move = moves[0]

        for move in self._get_ordered_moves(board):
            if move not in moves:
                continue

            # Simulate move
            board_copy = copy.deepcopy(board)
            board_copy.do_move(move)

            # Evaluate with minimax
            score = self._minimax(board_copy, self.max_depth - 1,
                                  -float('inf'), float('inf'), False)

            # Update best move
            if score > best_score:
                best_score = score
                best_move = move

        self.time_spent = time.time() - start_time
        return best_move

    def _check_critical_move(self, board):
        """Check for urgent defensive or winning positions"""
        opponent = 3 - self.player

        # Check for winning moves
        for move in board.availables:
            # Test if we can win
            board_copy = copy.deepcopy(board)
            board_copy.do_move(move)
            end, winner = board_copy.game_end()
            if end and winner == self.player:
                return move

        # Check opponent's winning moves
        for move in board.availables:
            board_copy = copy.deepcopy(board)
            old_player = board_copy.current_player
            board_copy.current_player = opponent
            board_copy.do_move(move)
            end, winner = board_copy.game_end()
            if end and winner == opponent:
                return move

        # Check for open four threats
        threat = self._find_threat(board, opponent, "open_four")
        if threat: return threat

        threat = self._find_threat(board, self.player, "open_four")
        if threat: return threat

        # Check for four threats
        threat = self._find_threat(board, opponent, "four")
        if threat: return threat

        threat = self._find_threat(board, self.player, "four")
        if threat: return threat

        # Check for open three threats
        threat = self._find_threat(board, opponent, "open_three")
        if threat: return threat

        threat = self._find_threat(board, self.player, "open_three")
        if threat: return threat

        return None

    def _find_threat(self, board, player, threat_type):
        """Find specific threats on board"""
        for move in board.availables:
            board_copy = copy.deepcopy(board)
            board_copy.states[move] = player
            patterns = self._get_patterns(board_copy, player)
            if patterns[threat_type] > 0:
                return move
            del board_copy.states[move]
        return None

    def _get_ordered_moves(self, board):
        """Get moves ordered by potential strength"""
        scored_moves = []

        for move in board.availables:
            h, w = board.move_to_location(move)
            score = 0

            # Score adjacent pieces
            for dh in [-1, 0, 1]:
                for dw in [-1, 0, 1]:
                    if dh == 0 and dw == 0:
                        continue
                    nh, nw = h + dh, w + dw
                    if 0 <= nh < board.height and 0 <= nw < board.width:
                        neighbor = board.location_to_move([nh, nw])
                        if neighbor in board.states:
                            score += 1
                            if board.states[neighbor] != self.player:
                                score += 1  # Prioritize defense

            # Prefer center
            center_h, center_w = board.height // 2, board.width // 2
            dist = abs(h - center_h) + abs(w - center_w)
            score += max(0, (board.width + board.height) // 2 - dist) // 2

            scored_moves.append((move, score))

        # Sort by score (descending)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves]

    def _minimax(self, board, depth, alpha, beta, is_maximizing):
        """Minimax algorithm with alpha-beta pruning"""
        self.nodes_evaluated += 1

        # Check terminal states
        end, winner = board.game_end()
        if end:
            if winner == self.player:  # Win
                return 10000 + depth
            elif winner == -1:  # Draw
                return 0
            else:  # Loss
                return -10000 - depth

        # Reached search depth limit
        if depth == 0:
            return self._evaluate(board)

        # Maximizing player (self)
        if is_maximizing:
            max_eval = -float('inf')
            moves = self._get_ordered_moves(board)

            for move in moves:
                if move not in board.availables:
                    continue

                board_copy = copy.deepcopy(board)
                board_copy.do_move(move)
                eval = self._minimax(board_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)

                # Alpha-Beta pruning
                if beta <= alpha:
                    break
            return max_eval

        # Minimizing player (opponent)
        else:
            min_eval = float('inf')
            moves = self._get_ordered_moves(board)

            for move in moves:
                if move not in board.availables:
                    continue

                board_copy = copy.deepcopy(board)
                board_copy.do_move(move)
                eval = self._minimax(board_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)

                # Alpha-Beta pruning
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate(self, board):
        """Evaluate board state"""
        opponent = 3 - self.player
        score = 0

        # Get patterns for both players
        patterns = self._get_patterns(board, self.player)
        opp_patterns = self._get_patterns(board, opponent)

        # Score patterns (offense)
        score += patterns['five'] * 100000
        score += patterns['open_four'] * 10000
        score += patterns['four'] * 1000
        score += patterns['open_three'] * 100
        score += patterns['three'] * 10
        score += patterns['open_two'] * 8
        score += patterns['two'] * 3

        # Score opponent patterns (defense - higher weights)
        score -= opp_patterns['five'] * 150000
        score -= opp_patterns['open_four'] * 15000
        score -= opp_patterns['four'] * 1500
        score -= opp_patterns['open_three'] * 150
        score -= opp_patterns['three'] * 20
        score -= opp_patterns['open_two'] * 10
        score -= opp_patterns['two'] * 5

        # Add position-based scoring
        for move, player in board.states.items():
            h, w = board.move_to_location(move)
            center_h, center_w = board.height // 2, board.width // 2
            dist = abs(h - center_h) + abs(w - center_w)
            position_value = max(0, 15 - dist) / 15.0

            if player == self.player:
                score += position_value * 10
            else:
                score -= position_value * 10

        return score

    def _get_patterns(self, board, player):
        """Get pattern counts from board"""
        patterns = {
            'five': 0,  # Five in a row
            'open_four': 0,  # Open four
            'four': 0,  # Four
            'open_three': 0,  # Open three
            'three': 0,  # Three
            'open_two': 0,  # Open two
            'two': 0  # Two in a row
        }

        # Create board state matrix
        board_state = np.zeros((board.height, board.width), dtype=int)
        for move, p in board.states.items():
            h, w = board.move_to_location(move)
            board_state[h][w] = p

        opponent = 3 - player

        # Convert board to string representation for pattern matching
        # Check horizontal lines
        for i in range(board.height):
            line = ""
            for j in range(board.width):
                if board_state[i][j] == player:
                    line += "1"
                elif board_state[i][j] == opponent:
                    line += "2"
                else:
                    line += "0"
            self._check_patterns(line, patterns)

        # Check vertical lines
        for j in range(board.width):
            line = ""
            for i in range(board.height):
                if board_state[i][j] == player:
                    line += "1"
                elif board_state[i][j] == opponent:
                    line += "2"
                else:
                    line += "0"
            self._check_patterns(line, patterns)

        # Check diagonal lines (both directions)
        # Main diagonal (top-left to bottom-right)
        for s in range(-(board.height - 1), board.width):
            line = ""
            for i in range(max(0, -s), min(board.height, board.width - s)):
                j = i + s
                if board_state[i][j] == player:
                    line += "1"
                elif board_state[i][j] == opponent:
                    line += "2"
                else:
                    line += "0"
            if len(line) >= 5:
                self._check_patterns(line, patterns)

        # Anti-diagonal (top-right to bottom-left)
        for s in range(board.width + board.height - 1):
            line = ""
            for i in range(max(0, s - board.width + 1), min(s + 1, board.height)):
                j = s - i
                if board_state[i][j] == player:
                    line += "1"
                elif board_state[i][j] == opponent:
                    line += "2"
                else:
                    line += "0"
            if len(line) >= 5:
                self._check_patterns(line, patterns)

        return patterns

    def _check_patterns(self, line, patterns):
        """Check line for various patterns"""
        # Five in a row
        patterns['five'] += line.count('11111')

        # Open four
        patterns['open_four'] += line.count('011110')

        # Four
        four_patterns = ['211110', '011112', '11110', '01111', '11101', '10111', '11011']
        for pattern in four_patterns:
            patterns['four'] += line.count(pattern)

        # Open three
        open_three_patterns = ['0011100', '001110', '010110', '011010']
        for pattern in open_three_patterns:
            patterns['open_three'] += line.count(pattern)

        # Three
        three_patterns = ['2011100', '0011102', '0111', '1110', '010101']
        for pattern in three_patterns:
            patterns['three'] += line.count(pattern)

        # Open two
        open_two_patterns = ['001100', '001010', '010100']
        for pattern in open_two_patterns:
            patterns['open_two'] += line.count(pattern)

        # Two
        two_patterns = ['0110', '011', '110', '11']
        for pattern in two_patterns:
            patterns['two'] += line.count(pattern)

    def __str__(self):
        return f"Minimax(depth={self.max_depth})"