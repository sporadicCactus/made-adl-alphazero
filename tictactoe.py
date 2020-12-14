import torch
import torch.multiprocessing as mp

import gym
import numpy as np
from matplotlib import pyplot as plt

from copy import deepcopy

import os


class TicTacToe(gym.Env):

    def __init__(self, n_rows, n_cols, n_win):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_win = n_win

        self.board = np.zeros((self.n_rows, self.n_cols), dtype=int)
        self.gameOver = False
        self.boardHash = None
        # ход первого игрока
        self.curTurn = 1
        self.emptySpaces = None

        self.reset()

    def setState(self, state_hash):
        for ind, char in enumerate(state_hash):
            cell = int(char) - 1
            self.board[ind // self.n_rows, ind % self.n_rows] = cell
        self.boardHash = state_hash
        self.curTurn = 1 if self.board.sum() % 2 == 0 else -1
        self.emptySpaces = None
        self.gameOver = False

    def getEmptySpaces(self):
        if self.emptySpaces is None:
            res = np.where(self.board == 0)
            self.emptySpaces = [ (i, j) for i,j in zip(res[0], res[1]) ]
        return self.emptySpaces

    def makeMove(self, player, i, j):
        self.board[i, j] = player
        self.emptySpaces = None
        self.boardHash = None

    def getHash(self):
        if self.boardHash is None:
            self.boardHash = ''.join(['%s' % (x+1) for x in self.board.reshape(self.n_rows * self.n_cols)])
        return self.boardHash

    def isTerminal(self):
        # проверим, не закончилась ли игра
        cur_marks, cur_p = np.where(self.board == self.curTurn), self.curTurn
        for i,j in zip(cur_marks[0], cur_marks[1]):
#             print((i,j))
            win = False
            if i <= self.n_rows - self.n_win:
                if np.all(self.board[i:i+self.n_win, j] == cur_p):
                    win = True
            if not win:
                if j <= self.n_cols - self.n_win:
                    if np.all(self.board[i,j:j+self.n_win] == cur_p):
                        win = True
            if not win:
                if i <= self.n_rows - self.n_win and j <= self.n_cols - self.n_win:
                    if np.all(np.array([ self.board[i+k,j+k] == cur_p for k in range(self.n_win) ])):
                        win = True
            if not win:
                if i <= self.n_rows - self.n_win and j >= self.n_win-1:
                    if np.all(np.array([ self.board[i+k,j-k] == cur_p for k in range(self.n_win) ])):
                        win = True
            if win:
                self.gameOver = True
                return self.curTurn

        if len(self.getEmptySpaces()) == 0:
            self.gameOver = True
            return 0

        self.gameOver = False
        return None

    def printBoard(self):
        for i in range(0, self.n_rows):
            print('----'*(self.n_cols)+'-')
            out = '| '
            for j in range(0, self.n_cols):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('----'*(self.n_cols)+'-')

    def getState(self):
        return (self.getHash(), self.getEmptySpaces(), self.curTurn)

    def action_from_int(self, action_int):
        return ( int(action_int / self.n_cols), int(action_int % self.n_cols))

    def int_from_action(self, action):
        return action[0] * self.n_cols + action[1]

    def step(self, action):
        if self.board[action[0], action[1]] != 0:
            return self.getState(), -10, True, {}
        self.makeMove(self.curTurn, action[0], action[1])
        reward = self.isTerminal()
        self.curTurn = -self.curTurn
        return self.getState(), 0 if reward is None else reward, reward is not None, {}

    def reset(self):
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=int)
        self.boardHash = None
        self.gameOver = False
        self.emptySpaces = None
        self.curTurn = 1


def plot_board(env, pi, showtext=True, verbose=True, fontq=20, fontx=60):
    '''Рисуем доску с оценками из стратегии pi'''
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    X, Y = np.meshgrid(np.arange(0, env.n_rows), np.arange(0, env.n_rows))
    Z = np.zeros((env.n_rows, env.n_cols)) + .01
    s, actions = env.getHash(), env.getEmptySpaces()
    if pi is not None and pi.knows(s):
        for i, a in enumerate(actions):
            Z[a[0], a[1]] = pi.get_Q(s)[tuple(a)]
    ax.set_xticks([])
    ax.set_yticks([])
    surf = ax.imshow(Z, cmap=plt.get_cmap('Accent', 10), vmin=-1, vmax=1)
    if showtext:
        for i,a in enumerate(actions):
            if pi is not None:
                Q = pi.get_Q(s)
                if Q is not None:
                    ax.text( a[1] , a[0] , "%.3f" % Q[tuple(a)], fontsize=fontq, horizontalalignment='center', verticalalignment='center', color="w" )
#             else:
#                 ax.text( a[1] , a[0] , "???", fontsize=fontq, horizontalalignment='center', verticalalignment='center', color="w" )
    for i in range(env.n_rows):
        for j in range(env.n_cols):
            if env.board[i, j] == -1:
                ax.text(j, i, "O", fontsize=fontx, horizontalalignment='center', verticalalignment='center', color="w" )
            if env.board[i, j] == 1:
                ax.text(j, i, "X", fontsize=fontx, horizontalalignment='center', verticalalignment='center', color="w" )
    cbar = plt.colorbar(surf, ticks=[0, 1])
    ax.grid(False)
    plt.show()


def get_and_print_move(env, pi, s, actions, random=False, verbose=True, fontq=20, fontx=60):
    '''Делаем ход, рисуем доску'''
    plot_board(env, pi, fontq=fontq, fontx=fontx)
    if verbose and (pi is not None):
        Q = pi.get_Q(s)
        if Q is not None:
            for i,a in enumerate(actions):
                print(i, a, Q[tuple(a)])
        else:
            print("Стратегия не знает, что делать...")
    if random:
        a = np.random.randint(len(actions))
        return actions[a]
    else:
        return pi.get_action_greedy(s)


def plot_test_game(env, pi1, pi2, random_crosses=False, random_naughts=True, verbose=True, fontq=20, fontx=60):
    '''Играем тестовую партию между стратегиями или со случайными ходами, рисуем ход игры'''
    done = False
    env.reset()
    while not done:
        s, actions = env.getHash(), env.getEmptySpaces()
        if env.curTurn == 1:
            a = get_and_print_move(env, pi1, s, actions, random=random_crosses, verbose=verbose, fontq=fontq, fontx=fontx)
        else:
            a = get_and_print_move(env, pi2, s, actions, random=random_naughts, verbose=verbose, fontq=fontq, fontx=fontx)
        observation, reward, done, info = env.step(a)
        if reward == 1:
            print("Крестики выиграли!")
            plot_board(env, None, showtext=False, fontq=fontq, fontx=fontx)
        if reward == -1:
            print("Нолики выиграли!")
            plot_board(env, None, showtext=False, fontq=fontq, fontx=fontx)


def play_episode(env, cross_player, naught_player):
    env.reset()
    cross_player.reset()
    naught_player.reset()
    done = False
    while not done:
        player = cross_player if env.curTurn == 1 else naught_player
        state_hash = env.getHash()
        action = player.get_action(state_hash)
        (new_state_hash, _, _), reward, done, _ = env.step(action)

    return reward

def play_many_episodes_worker(env, cross_player, naught_player, n_episodes, queue):
    torch.set_num_threads(1)
    rewards = [
        play_episode(env, cross_player, naught_player)\
        for _ in range(n_episodes)
    ]
    reward = np.array(rewards).mean()
    queue.put(reward)
    return reward

def match_players(env, player, other_player, n_games, n_workers=12):
    player = deepcopy(player).cpu().eval().share_memory()
    other_player = deepcopy(other_player).cpu().eval().share_memory()

    #with mp.Pool(n_workers) as pool:
    #    cross_rewards = pool.starmap(
    #        play_many_episodes,
    #        [(env, player, other_player, n_games//n_workers) for _ in range(n_workers)]
    #    )

    #    naught_rewards = pool.starmap(
    #        play_many_episodes,
    #        [(env, other_player, player, n_games//n_workers) for _ in range(n_workers)]
    #    )

    p_cross, p_naught = [], []
    q_cross, q_naught = mp.Queue(), mp.Queue()
    for _ in range(n_workers):
        p = mp.Process(
            target=play_many_episodes_worker,
            args=(env, player, other_player, max(n_games//n_workers, 1), q_cross)
        )
        p.start()
        p_cross.append(p)
        p = mp.Process(
            target=play_many_episodes_worker,
            args=(env, other_player, player, max(n_games//n_workers, 1), q_naught)
        )
        p.start()
        p_naught.append(p)

    for p in p_cross:
        p.join()
    for p in p_naught:
        p.join()

    cross_rewards, naught_rewards = [], []
    while not q_cross.empty():
        cross_rewards.append(q_cross.get())
    while not q_naught.empty():
        naught_rewards.append(q_naught.get())

    cross_reward = np.array(cross_rewards).mean()
    naught_reward = - np.array(naught_rewards).mean()

    return cross_reward, naught_reward
