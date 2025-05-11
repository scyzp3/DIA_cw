# -*- coding: utf-8 -*-
"""
AlphaZero style Monte Carlo Tree Search implementation,
using a policy-value network to guide tree search and evaluate leaf nodes
"""

import numpy as np
import copy


def softmax(x):
    """softmax function"""
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """Node in the MCTS tree"""

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # mapping from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children nodes"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select a child node that maximizes action value Q plus bonus u(P)
        Return: tuple(action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node value from leaf evaluation"""
        # Count visit times
        self._n_visits += 1
        # Update Q, average of all visits
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like update(), but applied recursively for all ancestors"""
        # If it's not the root node, update parent first
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value of this node
        c_puct: a parameter controlling the degree of exploration
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if the node is a leaf, i.e., has no expanded nodes"""
        return self._children == {}

    def is_root(self):
        """Check if the node is the root"""
        return self._parent is None


class MCTS(object):
    """AlphaZero style Monte Carlo Tree Search"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes a board state and outputs a list of (action, probability)
                         tuples and a score in [-1, 1] (i.e., the expected value of the end game score
                         from the current player's perspective)
        c_puct: a constant determining the level of exploration, higher values means relying more on prior
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run one playout from the root to the leaf, get the leaf value, and propagate it back through parent nodes
        state: must be a copy as the state will be modified in-place
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next step
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using policy network
        action_probs, leaf_value = self._policy(state)
        # Check for end of game
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # For end state, return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value for this traversal
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts and return the available actions and their corresponding probabilities
        state: the current game state
        temp: temperature parameter to control the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # Calculate the move probabilities based on the visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        # Calculate probabilities using softmax
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward, keep everything we know about the subtree"""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI Player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet noise for exploration (needed for self-play training only)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent to choosing the move with the highest probability
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("Warning: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)