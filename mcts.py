import random
from math import log, sqrt

from copy import deepcopy

class Node:
    def __init__(self, state_hash, available_actions, turn, probs=None, value=None, C=1.0):
        self.state_hash = state_hash
        self.Q = {action: 0. for action in available_actions}
        self.N = {action: 0 for action in available_actions}
        if probs is not None:
            self.P = {action: probs[action] for action in available_actions}
        else:
            self.P = None
        self.value = value
        self.turn = turn
        self.C = C
        
    def _compute_ucb_values(self):
        N_tot = sum(self.N.values())
        ucb_values = {
            action: self.Q[action] + self.C*sqrt(log(N_tot))/self.N[action]\
            if self.N[action] > 0 else float("inf")
            for action in self.Q.keys()
        } if self.P is None else {
            action: self.Q[action] + self.C*self.P[action]*sqrt(N_tot)/(1 + self.N[action])\
            for action in self.Q.keys()
        }
        return ucb_values
        
    def update(self, action, global_reward):
        self.N[action] += 1
        self.Q[action] += (self.turn*global_reward - self.Q[action])/self.N[action]
        
    def get_action(self):
        ucb_values = self._compute_ucb_values()
        best_value = max(ucb_values.values())
        best_actions = [action for action, value in ucb_values.items() if value==best_value]
        action = random.choice(best_actions)
        return action

class MCTSSession:

    def __init__(self, env, tree, exploration_parameter=1.0):
        self.env = deepcopy(env)
        self.tree = tree
        self.exploration_parameter = exploration_parameter
        self.done = False

    def update_tree(self):
        if self.done:
            raise Exception("This session is alredy done.")
        root_hash = self.env.getHash()
        self.trajectory = {
            "nodes": [self.tree[root_hash]],
            "actions": []
        }
        self._traverse_tree()
        self._rollout()
        self._backpropagate()

    def _initialize_node(self, state_hash, available_actions, turn):
        return Node(state_hash, available_actions, turn, C=self.exploration_parameter)

    def _traverse_tree(self):
        while True:
            last_node = self.trajectory["nodes"][-1]
            action = last_node.get_action()
            self.trajectory["actions"].append(action)
            (state_hash, available_actions, turn), reward, done, _ = self.env.step(action)
            self.done = done

            if self.done:
                self.episode_reward = reward
                break
            try:
                self.trajectory["nodes"].append(self.tree[state_hash])
            except:
                new_node = self._initialize_node(state_hash, available_actions, turn)
                self.tree[state_hash] = new_node
                self.trajectory["nodes"].append(new_node)
                break

    def _get_rollout_action(self, state_hash):
        return random.choice(state_hash)

    def _rollout(self):
        while not self.done:
            action = self._get_rollout_action(self.env.getEmptySpaces())
            if len(self.trajectory["actions"]) < len(self.trajectory["nodes"]):
                self.trajectory["actions"].append(action)
            _, reward, done, _ = self.env.step(action)
            self.done = done
            self.episode_reward = reward

    def _backpropagate(self):
        for node, action in zip(self.trajectory["nodes"], self.trajectory["actions"]):
            node.update(action, self.episode_reward)


class MCTSZeroSession(MCTSSession):
    def __init__(self, env, tree, player, exploration_parameter=1.0):
        super().__init__(env, tree, exploration_parameter)
        self.player = player

    def _initialize_node(self, state_hash, available_actions, turn):
        est = self.player.estimate(state_hash)
        node = Node(
            state_hash, available_actions, turn,
            probs=est["probs"], value=est["value"]
        )
        return node

    def _rollout(self):
        if len(self.trajectory["nodes"]) > len(self.trajectory["actions"]):
            node = self.trajectory["nodes"].pop(-1)
            self.episode_reward = node.value*node.turn


class MCTS:

    def __init__(self, num_simulations, exploration_parameter):
        self.num_simulations = num_simulations
        self.exploration_parameter = exploration_parameter
        self.cached_tree = None

    def _create_session(self, env, tree, player=None):
        session = MCTSSession(env, tree, self.exploration_parameter)
        return session

    def __call__(self, env, player=None):
        root_hash = env.getHash()
        if self.cached_tree is None:
            if hasattr(player, "estimate"):
                est = player.estimate(root_hash)
            else:
                est = {}
            root = Node(*env.getState(), **est)
            tree = {root_hash: root}
            self.cached_tree = tree
        else:
            tree = self.cached_tree
            try:
                root = tree[root_hash]
            except:
                if hasattr(player, "estimate"):
                    est = player.estimate(root_hash)
                else:
                    est = {}
                root = Node(*env.getState(), **est)
                tree[root_hash] = root
        for _ in range(self.num_simulations):
            session = self._create_session(env, tree, player)
            session.update_tree()
        return root

    def reset(self):
        self.cached_tree = None


class MCTSZero(MCTS):

    def __init__(self, num_simulations, exploration_parameter):
        super().__init__(num_simulations, exploration_parameter)

    def _create_session(self, env, tree, player):
        session = MCTSZeroSession(env, tree, player, self.exploration_parameter)
        return session
