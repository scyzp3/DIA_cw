# -*- coding: utf-8 -*-
"""
Optimized AlphaZero training script using PyTorch

@author: Optimized based on Junxiao Song's implementation
"""

from __future__ import print_function
import random
import numpy as np
import torch
import time
import os
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
import matplotlib.pyplot as plt


class TrainPipeline():
    def __init__(self, init_model=None, board_size=(6, 6), n_in_row=4, use_gpu=False):
        # Board parameters
        self.board_width, self.board_height = board_size
        self.n_in_row = n_in_row
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # Training parameters
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # Adaptive learning rate adjustment
        self.temp = 1.0  # Temperature parameter
        self.n_playout = 400  # Number of simulations per step
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # Batch size
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # Number of training steps per update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0

        # Pure MCTS simulations for evaluation
        self.pure_mcts_playout_num = 1000

        # GPU settings
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Load or create policy value network
        if init_model:
            # Start training from initial model
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model,
                                                   use_gpu=self.use_gpu)
        else:
            # Start training from a new policy value network
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   use_gpu=self.use_gpu)

        # Create MCTS player
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

        # Training records
        self.train_stats = {
            'loss': [],
            'entropy': [],
            'win_ratio': [],
            'episode_len': []
        }

    def get_equi_data(self, play_data):
        """
        Augment the dataset through rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # Rotate counter-clockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # Flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """Collect self-play data for training"""
        total_episode_len = 0
        for i in range(n_games):
            # Start self-play
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            # Record game length
            episode_len = len(play_data)
            total_episode_len += episode_len
            # Augment data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

        # Update average game length
        self.episode_len = total_episode_len / n_games
        self.train_stats['episode_len'].append(self.episode_len)

        return self.episode_len

    def policy_update(self):
        """Update the policy-value network"""
        # Sample batch data from buffer
        mini_batch = random.sample(self.data_buffer, min(self.batch_size, len(self.data_buffer)))
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        # Train for certain epochs
        total_loss = 0
        total_entropy = 0
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            total_loss += loss
            total_entropy += entropy

            # Get new probabilities and state values
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            # Calculate KL divergence
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )

            # If KL divergence is too large, stop training early
            if kl > self.kl_targ * 4:
                break

        # Record average loss and entropy
        avg_loss = total_loss / (i + 1)
        avg_entropy = total_entropy / (i + 1)
        self.train_stats['loss'].append(avg_loss)
        self.train_stats['entropy'].append(avg_entropy)

        # Adaptively adjust learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # Calculate explained variance
        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        avg_loss,
                        avg_entropy,
                        explained_var_old,
                        explained_var_new))

        return avg_loss, avg_entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against pure MCTS player
        Note: This is only for monitoring the training progress
        """
        # Create current MCTS player
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)

        # Create pure MCTS player as opponent
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)

        # Record win/loss results
        win_cnt = defaultdict(int)
        for i in range(n_games):
            # Alternate first and second player
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1

        # Calculate win ratio, considering draws
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games

        # Record win ratio
        self.train_stats['win_ratio'].append(win_ratio)

        print("Evaluation - Simulations:{}, Win: {}, Loss: {}, Draw:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))

        return win_ratio

    def save_training_stats(self, filename='training_stats.npz'):
        """Save training statistics"""
        np.savez(filename,
                 loss=np.array(self.train_stats['loss']),
                 entropy=np.array(self.train_stats['entropy']),
                 win_ratio=np.array(self.train_stats['win_ratio']),
                 episode_len=np.array(self.train_stats['episode_len']))
        print(f"Training statistics saved to: {filename}")

    def plot_training_stats(self, save_path=None):
        """Plot training statistics charts"""
        plt.figure(figsize=(15, 10))

        # Plot loss curve
        plt.subplot(2, 2, 1)
        plt.plot(self.train_stats['loss'])
        plt.title('Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')

        # Plot entropy curve
        plt.subplot(2, 2, 2)
        plt.plot(self.train_stats['entropy'])
        plt.title('Entropy')
        plt.xlabel('Training Steps')
        plt.ylabel('Entropy')

        # Plot win ratio curve
        plt.subplot(2, 2, 3)
        plt.plot(self.train_stats['win_ratio'])
        plt.title('Win Ratio')
        plt.xlabel('Evaluations')
        plt.ylabel('Win Ratio')

        # Plot episode length curve
        plt.subplot(2, 2, 4)
        plt.plot(self.train_stats['episode_len'])
        plt.title('Episode Length')
        plt.xlabel('Training Steps')
        plt.ylabel('Episode Length')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Training statistics chart saved to: {save_path}")
        else:
            plt.show()

    def run(self):
        """Run the training pipeline"""
        # Create model save directory
        if not os.path.exists('../models'):
            os.makedirs('../models')

        try:
            start_time = time.time()
            for i in range(self.game_batch_num):
                # Collect self-play data
                episode_len = self.collect_selfplay_data(self.play_batch_size)
                print("Batch i:{}, episode length:{}".format(i + 1, episode_len))

                # Update policy when there's enough data
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                # Periodically check performance of current model and save parameters
                if (i + 1) % self.check_freq == 0:
                    print("Current self-play batch: {}".format(i + 1))
                    # Evaluate current policy
                    win_ratio = self.policy_evaluate()
                    # Save current model
                    self.policy_value_net.save_model('./models/current_policy.model')

                    # If performance is better, save as best model
                    if win_ratio > self.best_win_ratio:
                        print("Found better policy!")
                        self.best_win_ratio = win_ratio
                        # Update best policy
                        self.policy_value_net.save_model('./models/best_policy.model')
                        # If win ratio reaches 100% and pure MCTS simulations < 5000, increase difficulty
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

                # Save training statistics every 100 batches
                if (i + 1) % 100 == 0:
                    self.save_training_stats()

                # Display training time
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Trained {i + 1} batches, time: {elapsed:.1f} seconds, average per batch: {elapsed / (i + 1):.2f} seconds")

        except KeyboardInterrupt:
            print('\n\rUser interrupted training')

        # Save final training statistics
        self.save_training_stats()

        # Plot training statistics chart
        self.plot_training_stats(save_path='training_stats.png')

        # Training completion report
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTraining completed! Total duration: {total_time:.1f} seconds")
        print(f"Best win ratio: {self.best_win_ratio:.3f}")
        print(f"Final pure MCTS simulation count: {self.pure_mcts_playout_num}")
        print(f"Training statistics saved to: training_stats.npz")
        print(f"Training statistics chart saved to: training_stats.png")
        print(f"Best model saved to: ./models/best_policy.model")


if __name__ == '__main__':
    # Set board size
    board_size = (8, 8)  # Width and height
    n_in_row = 5  # Number of pieces in a row to win

    # Set whether to use GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using GPU for training")
    else:
        print("Using CPU for training")

    # Load existing model to continue training (optional)
    init_model = None
    if os.path.exists('../models/best_policy.model'):
        init_model = './models/best_policy.model'
        print(f"Loading existing model: {init_model}")

    # Create training pipeline
    training_pipeline = TrainPipeline(init_model=init_model,
                                      board_size=board_size,
                                      n_in_row=n_in_row,
                                      use_gpu=use_gpu)

    # Start training
    print(f"Starting training, board size: {board_size}, {n_in_row} in a row")
    training_pipeline.run()