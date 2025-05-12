import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

# Constants
BOARD_SIZE = 11
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
L_1NUM = 242
L_2NUM = 242
L_3NUM = 182
L_4NUM = 121
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meta', 'variable_pt_11x11.pth')

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.manual_seed(1)
np.random.seed(1)


class GomokuNet(nn.Module):
    def __init__(self):
        super(GomokuNet, self).__init__()
        self.fc1 = nn.Linear(L_1NUM, L_1NUM)
        self.fc2 = nn.Linear(L_1NUM, L_2NUM)
        self.fc3 = nn.Linear(L_2NUM, L_3NUM)
        self.fc4 = nn.Linear(L_3NUM, L_4NUM)
        self.fc5 = nn.Linear(L_4NUM, BOARD_AREA)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class DQNAgent:
    def __init__(self, learning_rate=0.18, reward_decay=0.01, e_greedy=0.9,
                 memory_size=500, batch_size=100, e_greedy_increment=0.0001):
        # Basic parameters
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.epsilon = 1.0
        self.epsilon_increment = e_greedy_increment
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_step_counter = 0

        # Experience replay buffer
        self.memory = np.zeros((memory_size, L_1NUM * 2 + 2))
        self.memory_counter = 0

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create networks
        self.eval_net = GomokuNet().to(self.device)
        self.target_net = GomokuNet().to(self.device)

        # Optimizer
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=learning_rate)

        # Training history
        self.cost_history = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        a = np.reshape(a, [1, 1])
        r = np.reshape(r, [1, 1])
        s = np.reshape(s, [1, L_1NUM])
        s_ = np.reshape(s_, [1, L_1NUM])

        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, board, observation):
        if np.random.uniform() <= self.epsilon:  # Choose best action
            obs = np.reshape(observation, [1, L_1NUM])
            obs_tensor = torch.FloatTensor(obs).to(self.device)

            self.eval_net.eval()
            with torch.no_grad():
                q_values = self.eval_net(obs_tensor).cpu().numpy()[0]

            flat_board = board.flatten()
            empty_cells = np.where(flat_board == 0)[0]

            if len(empty_cells) == 0:
                return None

            valid_q_values = {pos: q_values[pos] for pos in empty_cells}
            action = max(valid_q_values, key=valid_q_values.get)
        else:  # Choose random action
            flat_board = board.flatten()
            empty_cells = np.where(flat_board == 0)[0]

            if len(empty_cells) == 0:
                return None

            action = np.random.choice(empty_cells)

        return action

    def learn(self, flag=1):
        if self.learn_step_counter % 50 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print("\nTarget network updated\n")

        batch_size = 1 if flag == 2 else self.batch_size

        if self.memory_counter > self.memory_size:
            sample_indices = np.random.choice(self.memory_size, batch_size)
        else:
            sample_indices = np.random.choice(self.memory_counter, batch_size)

        batch_memory = self.memory[sample_indices, :]

        batch_state = torch.FloatTensor(batch_memory[:, :L_1NUM]).to(self.device)
        batch_next_state = torch.FloatTensor(batch_memory[:, -L_1NUM:]).to(self.device)

        self.eval_net.train()
        self.target_net.eval()

        q_eval = self.eval_net(batch_state)
        with torch.no_grad():
            q_next = self.target_net(batch_next_state)

        q_target = q_eval.clone().detach()
        batch_actions = batch_memory[:, L_1NUM].astype(int)
        batch_rewards = batch_memory[:, L_1NUM + 1]

        for i in range(batch_size):
            q_target[i, batch_actions[i]] = batch_rewards[i] + self.gamma * torch.max(q_next[i])

        loss = F.mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        cost = loss.item()
        self.cost_history.append(cost)

        self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_increment)
        self.learn_step_counter += 1

        return cost

    def save_model(self):
        save_data = {
            'model_state': self.eval_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_counter': self.learn_step_counter
        }

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(save_data, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"No saved model found at {MODEL_PATH}")
            return False

        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            self.eval_net.load_state_dict(checkpoint['model_state'])
            self.target_net.load_state_dict(checkpoint['model_state'])

            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            if 'learn_counter' in checkpoint:
                self.learn_step_counter = checkpoint['learn_counter']

            print(f"Model loaded successfully from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.title('DQN Learning Curve')
        plt.show()


def transform_state(state):
    white_state = state[0, :BOARD_AREA]
    black_state = state[0, BOARD_AREA:]
    return np.hstack((black_state.reshape(1, -1), white_state.reshape(1, -1)))