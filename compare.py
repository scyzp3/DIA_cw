# -*- coding: utf-8 -*-
"""
Simple Gomoku AI Comparison: AlphaZero vs Minimax and AlphaZero vs Pure MCTS
"""

import time
import torch
import matplotlib.pyplot as plt

from mcts.game import Board, Game
from mcts.mcts_pure import MCTSPlayer as MCTS_Pure
from mcts.mcts_alphaZero import MCTSPlayer as MCTS_AlphaZero
from mcts.policy_value_net_pytorch import PolicyValueNet
from minmax import MinimaxPlayer


def load_alphazero(model_file, board_size=(8, 8)):
    """Load AlphaZero model"""
    width, height = board_size
    try:
        policy_value_net = PolicyValueNet(
            width, height,
            model_file=model_file,
            use_gpu=torch.cuda.is_available()
        )
        return policy_value_net
    except:
        return None


def compare_ai(model_file='models/best_policy.model', n_games=10):
    """Compare AlphaZero vs Minimax and AlphaZero vs Pure MCTS"""
    # Setup
    board_size = (8, 8)
    n_in_row = 5
    width, height = board_size

    # Create board
    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)

    # Load AlphaZero
    policy_net = load_alphazero(model_file, board_size)
    if policy_net is None:
        print("Failed to load AlphaZero model")
        return False

    # Create players
    alphazero = MCTS_AlphaZero(policy_net.policy_value_fn, c_puct=5, n_playout=400)
    minimax = MinimaxPlayer(max_depth=2)
    pure_mcts = MCTS_Pure(c_puct=5, n_playout=1000)

    ai_pairs = [
        (alphazero, minimax, "AlphaZero", "Minimax"),
        (alphazero, pure_mcts, "AlphaZero", "PureMCTS")
    ]

    all_results = {}

    # Compare each pair
    for ai1, ai2, name1, name2 in ai_pairs:
        print(f"\nComparing {name1} vs {name2} ({n_games} games)...")

        # Results tracking
        wins = {1: 0, 2: 0, 'tie': 0}
        times = {1: 0, 2: 0}
        moves = {1: 0, 2: 0}

        # Play games
        for i in range(n_games):
            # Alternate first player
            if i % 2 == 0:
                players = {1: ai1, 2: ai2}
                names = {1: name1, 2: name2}
                board.init_board(0)  # p1 first
            else:
                players = {1: ai2, 2: ai1}
                names = {1: name2, 2: name1}
                board.init_board(0)  # p1 first

            ai1.set_player_ind(1 if i % 2 == 0 else 2)
            ai2.set_player_ind(2 if i % 2 == 0 else 1)

            # Game loop
            while True:
                current = board.get_current_player()
                player = players[current]

                # Time the move
                start = time.time()
                move = player.get_action(board)
                move_time = time.time() - start

                # Record stats
                times[current] += move_time
                moves[current] += 1

                # Make move
                board.do_move(move)

                # Check if game ended
                end, winner = board.game_end()
                if end:
                    if winner == -1:
                        wins['tie'] += 1
                    else:
                        wins[winner] += 1
                    print(f"Game {i + 1}: {names[winner if winner != -1 else 1]} "
                          f"{'won' if winner != -1 else 'tied'}")
                    break

        # Calculate stats
        if i % 2 == 0:
            win1, win2 = wins[1], wins[2]
        else:
            win1, win2 = wins[2], wins[1]

        print(f"\nResults - {name1} vs {name2}:")
        print(f"{name1} wins: {win1} ({win1 / n_games * 100:.1f}%)")
        print(f"{name2} wins: {win2} ({win2 / n_games * 100:.1f}%)")
        print(f"Ties: {wins['tie']} ({wins['tie'] / n_games * 100:.1f}%)")

        all_results[f"{name1}_vs_{name2}"] = (win1, win2, wins['tie'])

    # Plot results
    plt.figure(figsize=(12, 6))

    # Set up plot
    x = ['AlphaZero vs Minimax', 'AlphaZero vs PureMCTS']
    alphazero_wins = [all_results['AlphaZero_vs_Minimax'][0],
                      all_results['AlphaZero_vs_PureMCTS'][0]]
    opponent_wins = [all_results['AlphaZero_vs_Minimax'][1],
                     all_results['AlphaZero_vs_PureMCTS'][1]]
    ties = [all_results['AlphaZero_vs_Minimax'][2],
            all_results['AlphaZero_vs_PureMCTS'][2]]

    # Plot bars
    x_pos = range(len(x))
    width = 0.25

    plt.bar([p - width for p in x_pos], alphazero_wins, width,
            color='blue', label='AlphaZero')
    plt.bar(x_pos, opponent_wins, width,
            color='red', label='Opponent')
    plt.bar([p + width for p in x_pos], ties, width,
            color='gray', label='Tie')

    # Add labels
    for i, v in enumerate(alphazero_wins):
        plt.text(i - width, v + 0.1, str(v), ha='center')

    for i, v in enumerate(opponent_wins):
        plt.text(i, v + 0.1, str(v), ha='center')

    for i, v in enumerate(ties):
        plt.text(i + width, v + 0.1, str(v), ha='center')

    # Add titles and labels
    plt.xlabel('Comparison')
    plt.ylabel('Number of wins')
    plt.title(f'AlphaZero Performance ({n_games} games per pair)')
    plt.xticks(x_pos, x)
    plt.legend()

    # Save and show
    plt.savefig('alphazero_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return True


if __name__ == '__main__':
    compare_ai(n_games=10)