import gzip
import pickle
import random
import pickletools
import numpy as np
from rlcollection.rlmutex import rlmutex
from collections import namedtuple, deque

__version__ = 0.009

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.__ready = False
        self.episodes_indexes = []

    def buffer_resize(self, new_capacity: int):
        if self.capacity != new_capacity:
            with rlmutex:
                new_memory = deque([], maxlen=new_capacity)
                if new_capacity > self.capacity or (self.capacity > new_capacity > self.__len__()):
                    new_memory.extend(self.memory)
                elif new_capacity < self.capacity and new_capacity <= self.__len__():
                    new_memory.extend(list(self.memory)[self.__len__() - new_capacity:])
                    self.episodes_indexes = self.__recalc_indexes(self.__len__() - new_capacity)
                self.memory.clear()
                self.memory = new_memory
                self.capacity = new_capacity
                self.ready = len(self.memory) == self.capacity

    def add(self, *args):
        """ Save a transition (state, action, next_state, reward) """
        with rlmutex:
            if self.episodes_indexes:
                self.__recalc_episodes_indexes(1)
            self.memory.append(Transition(*args))
            if not self.ready:
                self.ready = len(self.memory) == self.capacity

    def push(self, *args):
        """ Save a transition (state, action, next_state, reward) """
        self.add(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def extend(self, batch_list):
        with rlmutex:
            if self.episodes_indexes:
                self.__recalc_episodes_indexes(len(batch_list))
            self.memory.extend(batch_list)
            if not self.ready:
                self.ready = len(self.memory) == self.capacity

    def __recalc_indexes(self, idx_diff):
        indexes_arr = np.asarray(self.episodes_indexes)
        indexes_arr = indexes_arr - idx_diff
        indexes_arr = indexes_arr[np.all(indexes_arr > 0, axis=1)]
        new_indexes = [tuple(row) for row in indexes_arr]
        return new_indexes

    def __recalc_episodes_indexes(self, episode_length: int):
        if self.ready:
            self.episodes_indexes = self.__recalc_indexes(episode_length)
            self.episodes_indexes.append((len(self.memory) - episode_length, len(self.memory)))
        else:
            if len(self.memory) + episode_length > self.capacity:
                self.episodes_indexes = self.__recalc_indexes(
                    (len(self.memory) + episode_length) - self.capacity)
                self.episodes_indexes.append((self.capacity - episode_length, self.capacity))
            else:
                self.episodes_indexes.append((len(self.memory), len(self.memory) + episode_length))

    def extend_episode(self, episode_data):
        with rlmutex:
            self.__recalc_episodes_indexes(len(episode_data))
            self.memory.extend(episode_data)
            if not self.ready:
                self.ready = len(self.memory) == self.capacity

    def sample_episode(self):
        if self.episodes_indexes:
            with rlmutex:
                episode_index = random.sample(self.episodes_indexes, 1)[0]
                # print(episode_index)
                return [self.memory[idx] for idx in range(episode_index[0], episode_index[1])]
        else:
            return []

    def first_episode(self):
        if self.episodes_indexes:
            with rlmutex:
                episode_index = self.episodes_indexes[0]
                return [self.memory[idx] for idx in range(episode_index[0], episode_index[1])]
        else:
            return []

    def __len__(self):
        return len(self.memory)

    def clear(self):
        with rlmutex:
            self.memory.clear()
            self.ready = len(self.memory) == self.capacity

    @property
    def ready(self):
        return self.__ready

    @ready.setter
    def ready(self, value):
        self.__ready = value

    def save(self, path_filename):
        with gzip.open(path_filename, "wb") as f:
            with rlmutex:
                pickled = pickle.dumps([self.memory, self.episodes_indexes, self.capacity])
                optimized_pickle = pickletools.optimize(pickled)
                f.write(optimized_pickle)

    def load(self, path_filename):
        with rlmutex:
            with gzip.open(path_filename, 'rb') as f:
                p = pickle.Unpickler(f)
                self.memory, self.episodes_indexes, capacity = p.load()
            if self.capacity != capacity:
                self.buffer_resize(capacity)
            self.ready = True


if __name__ == '__main__':
    rb = ReplayBuffer(capacity=3)
    rb.push([1, 2, 3, 4], 2, [2, 3, 4, 5], -1)
    data = [[[2, 3, 4, 5], 2, [3, 4, 5, 6], -1],
            [[3, 4, 5, 6], 2, [4, 5, 6, 7], -1]]
    episode_data = [Transition(*element) for element in data]
    rb.extend_episode(episode_data)
    print(list(rb.memory))
    experience = rb.sample_episode()
    print(experience)
    batch = Transition(*zip(*experience))
    print(batch.next_state)
