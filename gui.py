# -*- coding: utf-8 -*-
import pygame
import sys
import time
import os
import torch
from mcts.game import Board, Game
from minmax import MinimaxPlayer
from mcts.mcts_alphaZero import MCTSPlayer as MCTS_AlphaZero
from mcts.policy_value_net_pytorch import PolicyValueNet

# Colors and parameters
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
BOARD_COLOR = (220, 179, 92)
GRID_COLOR = (0, 0, 0)
HIGHLIGHT_COLOR = (255, 0, 0)
INFO_COLOR = (50, 50, 50)
BUTTON_COLOR = (100, 100, 200)
BUTTON_HOVER_COLOR = (120, 120, 220)
BUTTON_TEXT_COLOR = (255, 255, 255)
CELL_SIZE = 40
STONE_RADIUS = CELL_SIZE // 2 - 2
BOARD_MARGIN = 40
INFO_WIDTH = 200
DEFAULT_MODEL = 'models/best_policy.model'


def check_model_exists(model_file):
    if model_file and os.path.exists(model_file):
        return True
    return False


class MinimalGame:
    def __init__(self, model_file=DEFAULT_MODEL):
        # Initialize board
        self.board = Board(width=8, height=8, n_in_row=5)
        self.game = Game(self.board)

        # Game state
        self.game_over = False
        self.winner = None
        self.human_first = True
        self.human_player = 1
        self.ai_player = 2
        self.current_player = 1
        self.last_move = None
        self.ai_type = 'minimax'
        self.model_file = model_file
        self.ai_thinking = False
        self.ai_time = 0
        self.ai_nodes = 0

        # Preload AlphaZero model
        self.policy_value_net = None
        if check_model_exists(model_file):
            try:
                self.policy_value_net = PolicyValueNet(
                    8, 8, model_file=model_file, use_gpu=torch.cuda.is_available()
                )
                self.ai_type = 'alphazero'
            except Exception:
                self.policy_value_net = None

        # Create AI player
        self.ai = self._create_ai_player()

        # Initialize Pygame
        pygame.init()
        self.screen_width = BOARD_MARGIN * 2 + CELL_SIZE * 8 + INFO_WIDTH
        self.screen_height = BOARD_MARGIN * 2 + CELL_SIZE * 8
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Gomoku")

        # Load fonts
        self.font = pygame.font.SysFont("Arial", 24)
        self.big_font = pygame.font.SysFont("Arial", 32)

        # Buttons
        self.buttons = {
            'restart': {
                'rect': pygame.Rect(self.screen_width - INFO_WIDTH + 20, 60, 140, 40),
                'text': "Restart",
                'action': self.restart_game
            },
            'switch': {
                'rect': pygame.Rect(self.screen_width - INFO_WIDTH + 20, 120, 140, 40),
                'text': "Switch Turn",
                'action': self.switch_player
            },
            'ai_type': {
                'rect': pygame.Rect(self.screen_width - INFO_WIDTH + 20, 180, 140, 40),
                'text': f"AI: {self.ai_type}",
                'action': self.switch_ai
            }
        }

        # Initialize game
        self.init_game()

    def _create_ai_player(self):
        if self.ai_type == 'alphazero' and self.policy_value_net is not None:
            return MCTS_AlphaZero(
                self.policy_value_net.policy_value_fn,
                c_puct=5,
                n_playout=400
            )
        else:
            self.ai_type = 'minimax'
            return MinimaxPlayer(max_depth=2)

    def init_game(self):
        self.board.init_board(0 if self.human_first else 1)
        self.game_over = False
        self.winner = None
        self.current_player = 1
        self.last_move = None
        self.ai_time = 0
        self.ai_nodes = 0

        if self.human_first:
            self.human_player = 1
            self.ai_player = 2
        else:
            self.human_player = 2
            self.ai_player = 1

        self.ai.set_player_ind(self.ai_player)
        self.buttons['ai_type']['text'] = f"AI: {self.ai_type}"

        if not self.human_first:
            self.ai_thinking = True

    def switch_ai(self):
        if self.ai_type == 'minimax':
            if self.policy_value_net is not None:
                self.ai_type = 'alphazero'
                self.ai = MCTS_AlphaZero(
                    self.policy_value_net.policy_value_fn,
                    c_puct=5,
                    n_playout=400
                )
        else:
            self.ai_type = 'minimax'
            self.ai = MinimaxPlayer(max_depth=2)

        self.buttons['ai_type']['text'] = f"AI: {self.ai_type}"
        self.ai.set_player_ind(self.ai_player)

    def restart_game(self):
        self.init_game()

    def switch_player(self):
        self.human_first = not self.human_first
        self.init_game()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()

                # Check button clicks
                for button_name, button in self.buttons.items():
                    if button['rect'].collidepoint(mouse_pos):
                        button['action']()
                        return

                # Handle board clicks
                if (not self.game_over and not self.ai_thinking and
                        self.current_player == self.human_player):
                    x, y = mouse_pos
                    grid_x = (x - BOARD_MARGIN) // CELL_SIZE
                    grid_y = (y - BOARD_MARGIN) // CELL_SIZE

                    if (0 <= grid_x < 8 and 0 <= grid_y < 8):
                        move = self.board.location_to_move([grid_y, grid_x])
                        if move in self.board.availables:
                            self.make_move(move)

    def make_move(self, move):
        self.board.do_move(move)
        self.last_move = move
        self.current_player = 3 - self.current_player

        self.check_game_end()

        if not self.game_over and self.current_player == self.ai_player:
            self.ai_thinking = True

    def check_game_end(self):
        end, winner = self.board.game_end()
        if end:
            self.game_over = True
            self.winner = winner

    def ai_move(self):
        if self.ai_thinking and not self.game_over:
            start_time = time.time()
            move = self.ai.get_action(self.board)
            self.ai_time = time.time() - start_time
            self.ai_nodes = getattr(self.ai, 'nodes_evaluated', 0)

            self.board.do_move(move)
            self.last_move = move
            self.current_player = 3 - self.current_player

            self.check_game_end()
            self.ai_thinking = False

    def draw_board(self):
        # Board background
        self.screen.fill(WHITE)
        board_rect = pygame.Rect(
            BOARD_MARGIN - 10,
            BOARD_MARGIN - 10,
            CELL_SIZE * 8 + 20,
            CELL_SIZE * 8 + 20
        )
        pygame.draw.rect(self.screen, BOARD_COLOR, board_rect)

        # Draw grid lines
        for i in range(9):
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN),
                (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN + 8 * CELL_SIZE),
                2 if i == 0 or i == 8 else 1
            )

        for i in range(9):
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (BOARD_MARGIN, BOARD_MARGIN + i * CELL_SIZE),
                (BOARD_MARGIN + 8 * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE),
                2 if i == 0 or i == 8 else 1
            )

        # Draw stones
        for move, player in self.board.states.items():
            y, x = self.board.move_to_location(move)

            stone_color = BLACK if player == 1 else WHITE
            border_color = WHITE if player == 1 else BLACK
            pygame.draw.circle(
                self.screen,
                stone_color,
                (BOARD_MARGIN + x * CELL_SIZE, BOARD_MARGIN + y * CELL_SIZE),
                STONE_RADIUS
            )
            pygame.draw.circle(
                self.screen,
                border_color,
                (BOARD_MARGIN + x * CELL_SIZE, BOARD_MARGIN + y * CELL_SIZE),
                STONE_RADIUS,
                1
            )

            # Mark last move
            if move == self.last_move:
                pygame.draw.circle(
                    self.screen,
                    HIGHLIGHT_COLOR,
                    (BOARD_MARGIN + x * CELL_SIZE, BOARD_MARGIN + y * CELL_SIZE),
                    STONE_RADIUS // 2,
                    2
                )

    def draw_info_panel(self):
        # Info panel background
        info_rect = pygame.Rect(
            self.screen_width - INFO_WIDTH,
            0,
            INFO_WIDTH,
            self.screen_height
        )
        pygame.draw.rect(self.screen, (240, 240, 240), info_rect)
        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (self.screen_width - INFO_WIDTH, 0),
            (self.screen_width - INFO_WIDTH, self.screen_height),
            2
        )

        # Game title
        title_text = self.big_font.render("Gomoku", True, INFO_COLOR)
        self.screen.blit(
            title_text,
            (self.screen_width - INFO_WIDTH + 20, 15)
        )

        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()
        for button_name, button in self.buttons.items():
            button_color = BUTTON_HOVER_COLOR if button['rect'].collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, button_color, button['rect'], border_radius=5)

            button_text = self.font.render(button['text'], True, BUTTON_TEXT_COLOR)
            text_rect = button_text.get_rect(center=button['rect'].center)
            self.screen.blit(button_text, text_rect)

        # Game info
        y_pos = 240

        if not self.game_over:
            current_text = "Current: "
            current_text += "You" if self.current_player == self.human_player else "AI"
            current_rendered = self.font.render(current_text, True, INFO_COLOR)
            self.screen.blit(current_rendered, (self.screen_width - INFO_WIDTH + 20, y_pos))

        y_pos += 40

        player_text = "You: {} ({})".format(
            "Black" if self.human_player == 1 else "White",
            "First" if self.human_first else "Second"
        )
        player_rendered = self.font.render(player_text, True, INFO_COLOR)
        self.screen.blit(player_rendered, (self.screen_width - INFO_WIDTH + 20, y_pos))

        y_pos += 30

        ai_text = "AI: {} ({})".format(
            "Black" if self.ai_player == 1 else "White",
            "First" if not self.human_first else "Second"
        )
        ai_rendered = self.font.render(ai_text, True, INFO_COLOR)
        self.screen.blit(ai_rendered, (self.screen_width - INFO_WIDTH + 20, y_pos))

        y_pos += 40

        # AI stats
        ai_type_text = f"AI Type: {self.ai_type}"
        ai_type_rendered = self.font.render(ai_type_text, True, INFO_COLOR)
        self.screen.blit(ai_type_rendered, (self.screen_width - INFO_WIDTH + 20, y_pos))

        y_pos += 30

        ai_time_text = f"Time: {self.ai_time:.3f}s"
        ai_time_rendered = self.font.render(ai_time_text, True, INFO_COLOR)
        self.screen.blit(ai_time_rendered, (self.screen_width - INFO_WIDTH + 20, y_pos))

        if self.ai_nodes > 0:
            y_pos += 30
            ai_nodes_text = f"Nodes: {self.ai_nodes}"
            ai_nodes_rendered = self.font.render(ai_nodes_text, True, INFO_COLOR)
            self.screen.blit(ai_nodes_rendered, (self.screen_width - INFO_WIDTH + 20, y_pos))

        y_pos += 40

        # Game result
        if self.game_over:
            if self.winner == self.human_player:
                result_text = "You won!"
                result_color = (0, 128, 0)
            elif self.winner == self.ai_player:
                result_text = "AI won"
                result_color = (192, 0, 0)
            else:
                result_text = "Draw"
                result_color = (128, 128, 128)

            result_rendered = self.big_font.render(result_text, True, result_color)
            self.screen.blit(result_rendered, (self.screen_width - INFO_WIDTH + 20, y_pos))

    def run(self):
        clock = pygame.time.Clock()

        while True:
            self.handle_events()

            if self.ai_thinking:
                self.ai_move()

            self.draw_board()
            self.draw_info_panel()
            pygame.display.update()

            clock.tick(30)


if __name__ == "__main__":
    game = MinimalGame()
    game.run()