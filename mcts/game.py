# -*- coding: utf-8 -*-
"""
Gomoku game logic and rules implementation
"""

from __future__ import print_function
import numpy as np


class Board(object):
    """Gomoku game board"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # Board state, key: position, value: piece type
        self.states = {}
        # Number of pieces needed in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # Player 1 and Player 2

    def init_board(self, start_player=0):
        """Initialize the board"""
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('Board width and height cannot be less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # First player
        # Store available moves
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        Convert 1D position to 2D location
        For example: on a 3x3 board, positions are:
        6 7 8
        3 4 5
        0 1 2
        Position 5 has 2D location (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        """Convert 2D location to 1D position"""
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        Return the board state from the current player's perspective
        State shape: 4*width*height
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # Mark the last move position
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # Indicate current player's color
        return square_state[:, ::-1, :]

    def do_move(self, move):
        """Execute a move"""
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        """Check if there is a winner"""
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        # Positions that have been moved
        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            # Check horizontal
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            # Check vertical
            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            # Check diagonal (bottom-left to top-right)
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            # Check diagonal (bottom-right to top-left)
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check if the game is over"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        """Get the current player"""
        return self.current_player

    def copy(self):
        """Return a copy of the board"""
        board = Board(width=self.width, height=self.height, n_in_row=self.n_in_row)
        board.states = self.states.copy()
        board.current_player = self.current_player
        board.availables = self.availables.copy()
        board.last_move = self.last_move
        return board


class Game(object):
    """Game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and display game information"""
        width = board.width
        height = board.height

        print("Player", player1, "uses X".rjust(3))
        print("Player", player2, "uses O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """Start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be 0 (player1 first) or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game over. Winner is", players[winner])
                    else:
                        print("Game over. It's a tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """Start a self-play game using the MCTS player, reuse the search tree, and store self-play data for training"""
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # Store data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # Execute move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # Calculate winners from the perspective of the current player for each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # Reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game over. Winner is player:", winner)
                    else:
                        print("Game over. It's a tie")
                return winner, zip(states, mcts_probs, winners_z)