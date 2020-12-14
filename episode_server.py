import torch

import torch.multiprocessing as mp
import time

from copy import deepcopy

BUFFER_SIZE_LIMIT = 256

def generate_episode(env, player, zero_mode=False):
    env.reset()
    player.reset()
    cross_hist, naught_hist = [], []
    turn_hist, state_hist, Q_hist = [], [], []
    done = False
    while not done:
        turn = env.curTurn
        state = env.getHash()
        action, Q = player.get_action_and_Q(state)
        (new_state, _, _), reward, done, _ = env.step(action)

        state_hist.append(state)
        Q_hist.append(Q)
        turn_hist.append(turn)
        
        hist = cross_hist if turn == 1 else naught_hist
        hist.append((state, action))
        hist.append(turn*reward)

    if not zero_mode:
        hist.append((None, None))
        hist = naught_hist if turn == 1 else cross_hist
        hist[-1] = - turn*reward
        hist.append((None, None))

        return cross_hist, naught_hist

    else:
        hist = [
            (state, reward*turn, prob) for state, prob, turn in zip(state_hist, Q_hist, turn_hist)
        ]

        return hist

def generate_episodes_in_loop(env, player, pipe, zero_mode=False):
    torch.set_num_threads(1)
    local_buffer = []
    while True:
        if pipe.poll():
            message = pipe.recv()
            if message in "dump":
                while len(local_buffer) > 0:
                    pipe.send(local_buffer.pop())
                pipe.send("done")
            if message == "kill":
                return

        if len(local_buffer) > BUFFER_SIZE_LIMIT:
            time.sleep(0.1)
            continue

        if not zero_mode:
            cross_hist, naught_hist = generate_episode(env, player, zero_mode)
            local_buffer.append(cross_hist)
            local_buffer.append(naught_hist)
        else:
            hist = generate_episode(env, player, zero_mode)
            local_buffer.append(hist)


class EpisodeServer:
    def __init__(self, env, player, num_workers=1, max_buffer_size=0, zero_mode=False):
        self.env = env
        self.player = deepcopy(player).cpu().eval().share_memory()
        self.num_workers = num_workers
        self.buffer = []
        self.workers = []
        self.pipes = []
        self.zero_mode = zero_mode

    def _run_workers(self):
        assert len(self.workers) == 0
        for _ in range(self.num_workers):
            pipe_local, pipe_remote = mp.Pipe()
            p = mp.Process(
                    target=generate_episodes_in_loop,
                    args=(self.env, self.player, pipe_remote, self.zero_mode)
            )
            p.start()
            self.workers.append(p)
            self.pipes.append(pipe_local)

    def _kill_workers(self):
        for pipe in self.pipes:
            pipe.send("kill")
        for pipe in self.pipes:
            while pipe.poll():
                message = pipe.recv()
                if not isinstance(message, str):
                    self.buffer.append(message)
        for p in self.workers:
            p.join()
        for p in self.workers:
            p.close()
        self.workers.clear()
        self.pipes.clear()

    def start(self):
        self.player.share_memory()
        self._run_workers()

    def stop(self):
        self._kill_workers()

    def update_player(self, new_player):
        self._kill_workers()
        self.player = deepcopy(new_player).cpu().eval().share_memory()
        self._run_workers()

    def get_episodes(self):
        for pipe in self.pipes:
            pipe.send("dump")
        for pipe in self.pipes:
            while True:
                message = pipe.recv()
                if message == "done":
                    break
                self.buffer.append(message)
        episodes = self.buffer
        self.buffer = []
        return episodes
