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
    def __init__(self):
        self.n_actions = BOARD_AREA
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.done = False
        self.num_white = 0
        self.num_black = 0

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.done = False
        return np.copy(self.board)

    def step(self, action, player):
        prev_reward = self._calculate_reward(player)

        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        if player == 'White':
            self.board[row, col] = 1
        else:
            self.board[row, col] = 2

        curr_reward = self._calculate_reward(player)
        reward = curr_reward - prev_reward
        state_repr = self.get_state_representation()

        return np.copy(self.board), state_repr, reward, self.done

    def get_state_representation(self):
        white_pieces = np.zeros((1, BOARD_AREA))
        black_pieces = np.zeros((1, BOARD_AREA))

        flat_board = self.board.flatten()
        white_positions = np.where(flat_board == 1)[0]
        black_positions = np.where(flat_board == 2)[0]

        if len(white_positions) > 0:
            white_pieces[0, white_positions] = 1
        if len(black_positions) > 0:
            black_pieces[0, black_positions] = 1

        return np.hstack((white_pieces, black_pieces))

    def _calculate_reward(self, player):
        self.num_white = 0
        self.num_black = 0
        self._check_all_patterns()

        if player == 'White':
            return self.num_white - self.num_black
        else:
            return self.num_black - self.num_white

    def _check_all_patterns(self):
        # Horizontal patterns
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE - 4):
                pattern = [self.board[row, col+i] for i in range(5)]
                self._match_pattern(pattern)

            for col in range(BOARD_SIZE - 5):
                pattern = [self.board[row, col+i] for i in range(6)]
                self._match_pattern(pattern)

        # Vertical patterns
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE - 4):
                pattern = [self.board[row+i, col] for i in range(5)]
                self._match_pattern(pattern)

            for row in range(BOARD_SIZE - 5):
                pattern = [self.board[row+i, col] for i in range(6)]
                self._match_pattern(pattern)

        # Diagonal patterns (top-left to bottom-right)
        for row in range(BOARD_SIZE - 4):
            for col in range(BOARD_SIZE - 4):
                pattern = [self.board[row+i, col+i] for i in range(5)]
                self._match_pattern(pattern)

            if row < BOARD_SIZE - 5:
                for col in range(BOARD_SIZE - 5):
                    pattern = [self.board[row+i, col+i] for i in range(6)]
                    self._match_pattern(pattern)

        # Diagonal patterns (top-right to bottom-left)
        for row in range(BOARD_SIZE - 4):
            for col in range(4, BOARD_SIZE):
                pattern = [self.board[row+i, col-i] for i in range(5)]
                self._match_pattern(pattern)

            if row < BOARD_SIZE - 5:
                for col in range(5, BOARD_SIZE):
                    pattern = [self.board[row+i, col-i] for i in range(6)]
                    self._match_pattern(pattern)

    def _match_pattern(self, pattern):
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
    def __init__(self):
        self.env = GomokuEnvironment()
        self.agent = DQNAgent()
        self.agent.load_model()

    def print_board(self):
        print("  ", end="")
        for col in range(BOARD_SIZE):
            print(f" {col+1:2}", end="")
        print()

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

                if self.env.board[row, col] != 0:
                    print("Position already occupied! Try again.")
                    continue

                return action

            except ValueError:
                print("Invalid input! Please enter numbers.")

    def ai_turn(self, player_side):
        print(f"AI's turn ({player_side})...")

        board_state = np.copy(self.env.board)
        observation = self.env.get_state_representation()
        action = self.agent.choose_action(board_state, observation)

        if action is None or self.env.board[action // BOARD_SIZE, action % BOARD_SIZE] != 0:
            empty_cells = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                         if self.env.board[r, c] == 0]
            if not empty_cells:
                return None

            row, col = empty_cells[np.random.choice(len(empty_cells))]
            action = row * BOARD_SIZE + col

        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        print(f"AI places at ({row+1}, {col+1})")

        return action

    def play_game(self):
        self.env.reset()

        while True:
            choice = input("Do you want to play as Black (X) or White (O)? (B/W): ").upper()
            if choice in ['B', 'W']:
                break
            print("Invalid choice. Please enter 'B' for Black or 'W' for White.")

        human_is_black = (choice == 'B')
        current_player = 'Black'
        game_over = False

        print("\nGame started! Black (X) goes first.")
        self.print_board()

        while not game_over:
            human_turn = (current_player == 'Black' and human_is_black) or \
                         (current_player == 'White' and not human_is_black)

            if human_turn:
                action = self.human_turn(current_player)
            else:
                action = self.ai_turn(current_player)

                if action is None:
                    print("Board is full. It's a draw!")
                    break

            _, _, reward, done = self.env.step(action, current_player)
            self.print_board()

            if done:
                winner = current_player
                print(f"Game over! {winner} wins!")

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

            current_player = 'White' if current_player == 'Black' else 'Black'

        again = input("Play again? (Y/N): ").upper()
        if again == 'Y':
            self.play_game()
        else:
            print("Thanks for playing!")


def play():
    print("======== Gomoku AI ========")
    print("You'll play against an AI trained with Deep Q-Learning.")
    game = GomokuGame()
    game.play_game()


if __name__ == "__main__":
    play()