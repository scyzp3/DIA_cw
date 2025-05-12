import numpy as np
import time
import argparse
from gomoku import GomokuEnvironment
from dqn import DQNAgent, transform_state


def self_play_training(episodes=100):
    env = GomokuEnvironment()
    agent = DQNAgent()
    agent.load_model()

    episode_lengths = []
    start_time = time.time()
    step = 0

    try:
        for episode in range(episodes):
            board = env.reset()
            observation = env.get_state_representation()

            done = False
            is_white_turn = True
            episode_step = 0

            print(f"Episode {episode+1}/{episodes}")

            while not done:
                if is_white_turn:
                    action = agent.choose_action(board, observation)
                    board, observation_, reward, done = env.step(action, 'White')
                    agent.store_transition(observation, action, reward, observation_)
                else:
                    transformed_state = transform_state(observation)
                    action = agent.choose_action(board, transformed_state)
                    board, observation_, reward, done = env.step(action, 'Black')
                    transformed_next = transform_state(observation_)
                    agent.store_transition(transformed_state, action, reward, transformed_next)

                if step > 200 and step % 100 == 0:
                    agent.learn()

                observation = observation_
                is_white_turn = not is_white_turn
                episode_step += 1
                step += 1

                if done:
                    break

            episode_lengths.append(episode_step)

            elapsed = time.time() - start_time
            print(f"Episode {episode+1} took {episode_step} steps")
            print(f"Total time: {elapsed:.1f}s, Steps/sec: {step/elapsed:.1f}")

            if (episode + 1) % 10 == 0:
                agent.save_model()
                print(f"Model saved at episode {episode+1}")

        agent.save_model()
        print("Training completed!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save_model()

    agent.plot_cost()


def evaluate_agent(episodes=10):
    env = GomokuEnvironment()
    agent = DQNAgent()

    if not agent.load_model():
        print("No trained model found. Train first.")
        return

    wins = 0
    losses = 0
    draws = 0

    for episode in range(episodes):
        board = env.reset()
        observation = env.get_state_representation()

        done = False
        is_white_turn = True

        print(f"Evaluation game {episode+1}/{episodes}")

        while not done:
            if is_white_turn:
                action = agent.choose_action(board, observation)
                player = 'White'
            else:
                flat_board = board.flatten()
                empty_cells = np.where(flat_board == 0)[0]
                if len(empty_cells) == 0:
                    break
                action = np.random.choice(empty_cells)
                player = 'Black'

            board, observation, reward, done = env.step(action, player)
            is_white_turn = not is_white_turn

            if done:
                if player == 'White':
                    wins += 1
                    print("Agent won!")
                else:
                    losses += 1
                    print("Random player won!")
                break

        if not done:
            draws += 1
            print("Draw!")

    win_rate = wins / episodes
    print(f"\nEvaluation results after {episodes} games:")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Win rate: {win_rate:.2f} ({int(win_rate*100)}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or evaluate Gomoku AI')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode to run: train or eval')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to run')

    args = parser.parse_args()

    if args.mode == 'train':
        self_play_training(episodes=args.episodes)
    else:
        evaluate_agent(episodes=args.episodes)