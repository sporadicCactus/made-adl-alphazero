import torch

import torch.multiprocessing as mp
import time

from copy import deepcopy

BUFFER_SIZE_LIMIT = 256

def generate_episode(env, player):
    env.reset()
    cross_hist, naught_hist = [], []
    done = False
    while not done:
        turn = env.curTurn
        state = env.getHash()
        action = player.get_action(state)
        (new_state, _, _), reward, done, _ = env.step(action)
        
        hist = cross_hist if turn == 1 else naught_hist
        hist.append((state, action))
        hist.append(turn*reward)

    hist.append((None, None))
    hist = naught_hist if turn == 1 else cross_hist
    hist[-1] = - turn*reward
    hist.append((None, None))

    return cross_hist, naught_hist


def generate_episodes_in_loop(env, player, message_pipe, episodes_buffer):
    torch.set_num_threads(1)
    while True:
        if message_pipe.poll():
            message = message_pipe.recv()
            if message == "kill":
                return

        if episodes_buffer.qsize() > BUFFER_SIZE_LIMIT:
            time.sleep(0.1)
            continue

        cross_hist, naught_hist = generate_episode(env, player)
        episodes_buffer.put(cross_hist)
        episodes_buffer.put(naught_hist)


class EpisodeServer:
    def __init__(self, env, player, num_workers=1, max_buffer_size=0):
        self.env = env
        self.player = deepcopy(player).cpu().eval()
        self.local_buffer = []
        self.buffer = mp.Queue(max_buffer_size)
        self.num_workers = num_workers
        self.workers = []
        self.pipes = []

    def _run_workers(self):
        assert len(self.workers) == 0
        for _ in range(self.num_workers):
            pipe_local, pipe_remote = mp.Pipe()
            p = mp.Process(
                    target=generate_episodes_in_loop,
                    args=(self.env, self.player, pipe_remote, self.buffer)
                )
            p.start()
            self.workers.append(p)
            self.pipes.append(pipe_local)

    def _empty_buffer(self):
        while not self.buffer.empty():
            self.local_buffer.append(self.buffer.get())

    def _kill_workers(self):
        for pipe in self.pipes:
            pipe.send("kill")
        self._empty_buffer()
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
        self.player = deepcopy(new_player).cpu().eval()
        self.player.share_memory()
        self._run_workers()

    def get_episodes(self):
        self._empty_buffer()
        episodes = self.local_buffer
        self.local_buffer = []
        return episodes
