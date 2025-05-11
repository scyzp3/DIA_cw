"""
Simple Gomoku game and environment - console version.
"""

import numpy as np
from dqn import DQNAgent, transform_state, BOARD_SIZE, BOARD_AREA

# Pattern evaluation values
VALUE1 = 50000  # *****
VALUE2 = 4320   # +****+
VALUE3 = 720    # -****+
VALUE4 = 720    # +***+
VALUE5 = 720    # -***++
VALUE6 = 120    # ++**+
VALUE10 = 1000  # Other patterns


class GomokuEnvironment:
    """Gomoku game environment."""

    def __init__(self):
        """Initialize the environment."""
        self.n_actions = BOARD_AREA
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.done = False
        self.num_white = 0  # For pattern evaluation
        self.num_black = 0  # For pattern evaluation

    def reset(self):
        """Reset the game board."""
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.done = False
        return np.copy(self.board)

    def step(self, action, player):
        """Take a step in the environment by placing a piece."""
        # Calculate reward before making the move
        prev_reward = self._calculate_reward(player)

        # Make the move
        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        if player == 'White':
            self.board[row, col] = 1
        else:
            self.board[row, col] = 2

        # Calculate reward after making the move
        curr_reward = self._calculate_reward(player)

        # The reward is the improvement in position
        reward = curr_reward - prev_reward

        # Create state representation
        state_repr = self.get_state_representation()

        return np.copy(self.board), state_repr, reward, self.done

    def get_state_representation(self):
        """Convert the board state to neural network input format."""
        # Create empty arrays for each player
        white_pieces = np.zeros((1, BOARD_AREA))
        black_pieces = np.zeros((1, BOARD_AREA))

        # Flatten the board
        flat_board = self.board.flatten()

        # Mark positions
        white_positions = np.where(flat_board == 1)[0]
        black_positions = np.where(flat_board == 2)[0]

        if len(white_positions) > 0:
            white_pieces[0, white_positions] = 1
        if len(black_positions) > 0:
            black_pieces[0, black_positions] = 1

        # Concatenate into state representation
        state_repr = np.hstack((white_pieces, black_pieces))

        return state_repr

    def _calculate_reward(self, player):
        """Calculate the reward for the current player."""
        self.num_white = 0
        self.num_black = 0

        # Calculate patterns in all directions
        self._check_all_patterns()

        # Return the relative advantage
        if player == 'White':
            return self.num_white - self.num_black
        else:
            return self.num_black - self.num_white

    def _check_all_patterns(self):
        """Check all possible patterns on the board."""
        # Horizontal patterns
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE - 4):
                pattern = [self.board[row, col+i] for i in range(5)]
                self._match_pattern(pattern)

            # Length 6 patterns
            for col in range(BOARD_SIZE - 5):
                pattern = [self.board[row, col+i] for i in range(6)]
                self._match_pattern(pattern)

        # Vertical patterns
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE - 4):
                pattern = [self.board[row+i, col] for i in range(5)]
                self._match_pattern(pattern)

            # Length 6 patterns
            for row in range(BOARD_SIZE - 5):
                pattern = [self.board[row+i, col] for i in range(6)]
                self._match_pattern(pattern)

        # Diagonal patterns (top-left to bottom-right)
        for row in range(BOARD_SIZE - 4):
            for col in range(BOARD_SIZE - 4):
                pattern = [self.board[row+i, col+i] for i in range(5)]
                self._match_pattern(pattern)

            # Length 6 patterns
            if row < BOARD_SIZE - 5:
                for col in range(BOARD_SIZE - 5):
                    pattern = [self.board[row+i, col+i] for i in range(6)]
                    self._match_pattern(pattern)

        # Diagonal patterns (top-right to bottom-left)
        for row in range(BOARD_SIZE - 4):
            for col in range(4, BOARD_SIZE):
                pattern = [self.board[row+i, col-i] for i in range(5)]
                self._match_pattern(pattern)

            # Length 6 patterns
            if row < BOARD_SIZE - 5:
                for col in range(5, BOARD_SIZE):
                    pattern = [self.board[row+i, col-i] for i in range(6)]
                    self._match_pattern(pattern)

    def _match_pattern(self, pattern):
        """Match pattern against known valuable configurations."""
        if len(pattern) == 5:
            # Five in a row
            if pattern == [1, 1, 1, 1, 1]:  # White
                self.done = True
                self.num_white += VALUE1
            elif pattern == [2, 2, 2, 2, 2]:  # Black
                self.done = True
                self.num_black += VALUE1

            # Open three patterns
            elif pattern in [[0, 1, 1, 1, 0], [0, 1, 1, 0, 1], [1, 0, 1, 1, 0],
                           [1, 1, 1, 0, 0], [0, 0, 1, 1, 1]]:
                self.num_white += VALUE4
            elif pattern in [[0, 2, 2, 2, 0], [0, 2, 2, 0, 2], [2, 0, 2, 2, 0],
                           [2, 2, 2, 0, 0], [0, 0, 2, 2, 2]]:
                self.num_black += VALUE4

            # Split four patterns
            elif pattern in [[1, 1, 0, 1, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 1],
                           [1, 1, 1, 1, 0], [0, 1, 1, 1, 1]]:
                self.num_white += VALUE10
            elif pattern in [[2, 2, 0, 2, 2], [2, 0, 2, 2, 2], [2, 2, 2, 0, 2],
                           [2, 2, 2, 2, 0], [0, 2, 2, 2, 2]]:
                self.num_black += VALUE10

            # Other patterns
            elif pattern in [[0, 0, 1, 1, 0], [0, 1, 1, 0, 0], [1, 1, 0, 0, 0]]:  # Open two
                self.num_white += VALUE6
            elif pattern in [[0, 0, 2, 2, 0], [0, 2, 2, 0, 0], [2, 2, 0, 0, 0]]:
                self.num_black += VALUE6

        elif len(pattern) == 6:
            # Open four
            if pattern == [0, 1, 1, 1, 1, 0]:
                self.num_white += VALUE2
            elif pattern == [0, 2, 2, 2, 2, 0]:
                self.num_black += VALUE2

            # Four with one end blocked
            elif pattern in [[2, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 2]]:
                self.num_white += VALUE3
            elif pattern in [[1, 2, 2, 2, 2, 0], [0, 2, 2, 2, 2, 1]]:
                self.num_black += VALUE3


class GomokuGame:
    """Console-based Gomoku game."""

    def __init__(self):
        """Initialize the game."""
        self.env = GomokuEnvironment()
        self.agent = DQNAgent()
        self.agent.load_model()

    def print_board(self):
        """Print the current board state."""
        # Print column headers
        print("  ", end="")
        for col in range(BOARD_SIZE):
            print(f" {col+1:2}", end="")
        print()

        # Print board with row headers
        for row in range(BOARD_SIZE):
            print(f"{row+1:2}", end=" ")
            for col in range(BOARD_SIZE):
                cell = self.env.board[row, col]
                if cell == 0:
                    print(" .", end=" ")
                elif cell == 1:
                    print(" O", end=" ")  # White
                else:
                    print(" X", end=" ")  # Black
            print()
        print()

    def human_turn(self, player_side):
        """Handle human player turn."""
        while True:
            try:
                print(f"Your turn ({player_side}).")
                row_input = input("Enter row (1-11): ")
                col_input = input("Enter column (1-11): ")

                row = int(row_input) - 1
                col = int(col_input) - 1

                if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                    print("Invalid position! Row and column must be between 1 and 11.")
                    continue

                action = row * BOARD_SIZE + col

                # Check if position is already occupied
                if self.env.board[row, col] != 0:
                    print("Position already occupied! Try again.")
                    continue

                return action

            except ValueError:
                print("Invalid input! Please enter numbers.")

    def ai_turn(self, player_side):
        """Handle AI turn."""
        print(f"AI's turn ({player_side})...")

        # Get current state
        board_state = np.copy(self.env.board)
        observation = self.env.get_state_representation()

        # Get AI action
        action = self.agent.choose_action(board_state, observation)

        # If no valid action or action is occupied, choose random empty cell
        if action is None or self.env.board[action // BOARD_SIZE, action % BOARD_SIZE] != 0:
            empty_cells = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                         if self.env.board[r, c] == 0]
            if not empty_cells:
                return None  # No valid moves

            row, col = empty_cells[np.random.choice(len(empty_cells))]
            action = row * BOARD_SIZE + col

        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        print(f"AI places at ({row+1}, {col+1})")

        return action

    def play_game(self):
        """Play a game of Gomoku."""
        # Reset game
        self.env.reset()

        # Ask player for side
        while True:
            choice = input("Do you want to play as Black (X) or White (O)? (B/W): ").upper()
            if choice in ['B', 'W']:
                break
            print("Invalid choice. Please enter 'B' for Black or 'W' for White.")

        human_is_black = (choice == 'B')

        # Game loop
        current_player = 'Black'  # Black goes first
        game_over = False

        print("\nGame started! Black (X) goes first.")
        self.print_board()

        while not game_over:
            # Determine whose turn it is
            human_turn = (current_player == 'Black' and human_is_black) or \
                         (current_player == 'White' and not human_is_black)

            # Get action based on whose turn it is
            if human_turn:
                action = self.human_turn(current_player)
            else:
                action = self.ai_turn(current_player)

                # Check for draw
                if action is None:
                    print("Board is full. It's a draw!")
                    break

            # Make move
            _, _, reward, done = self.env.step(action, current_player)

            # Print updated board
            self.print_board()

            # Check for game end
            if done:
                winner = current_player
                print(f"Game over! {winner} wins!")

                # Learning from human win
                if human_turn:
                    try:
                        print("AI is learning from your winning strategy...")
                        state = self.env.get_state_representation()
                        if human_is_black:
                            state = transform_state(state)
                        self.agent.learn(flag=2)
                        self.agent.save_model()
                        print("AI model updated.")
                    except Exception as e:
                        print(f"Error learning from human move: {e}")

                game_over = True
                break

            # Switch player
            current_player = 'White' if current_player == 'Black' else 'Black'

        # Ask to play again
        again = input("Play again? (Y/N): ").upper()
        if again == 'Y':
            self.play_game()
        else:
            print("Thanks for playing!")


def play():
    """Start a Gomoku game."""
    print("======== Gomoku AI ========")
    print("You'll play against an AI trained with Deep Q-Learning.")
    game = GomokuGame()
    game.play_game()


if __name__ == "__main__":
    play()