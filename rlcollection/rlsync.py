import threading
from multiprocessing.managers import SyncManager
from rlcollection.replaybuffer import MpReplayBuffer

ThLock = threading.Lock()

__version__ = 0.0013


class RlSync(SyncManager):
    def __init__(self, *args):
        super().__init__()

        self.register('MpReplayBuffer', MpReplayBuffer, exposed=['__len__', 'push', 'add',
                                                                 'sample', 'get', 'buffer_resize'])
        self.start()
        self.total_time_steps = self.Value('i', 0)
        self.time_step = self.Value('i', 0)
        self.episodes_rewards = self.list()
        self.episodes_length = self.list()
        self.memory = self.MpReplayBuffer()
        self.net_weights = self.dict()
        self.lock = self.Lock()

    def add_time_step(self):
        with self.lock:
            self.time_step.value += 1

    def get_time_step(self):
        return self.time_step.value

    def set_total_time_steps(self, value):
        with self.lock:
            self.total_time_steps.value = value

    def get_total_time_steps(self):
        return self.total_time_steps.value

    def append_episodes_rewards(self, value):
        with self.lock:
            self.episodes_rewards.append(value)

    def append_episodes_length(self, value):
        with self.lock:
            self.episodes_length.append(value)

    def set_net_weights(self, weights_dict):
        with self.lock:
            self.net_weights = weights_dict

    def get_net_weights(self):
        with self.lock:
            return self.net_weights


RLSYNC_obj = RlSync()
