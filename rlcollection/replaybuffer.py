import copy
import random
import threading
from collections import namedtuple, deque

from multiprocessing import Lock as MpLock
from multiprocessing.managers import SyncManager

threads_lock = threading.Lock()

__version__ = 0.007

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ThReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def buffer_resize(self, new_capacity: int):
        if self.capacity != new_capacity:
            with threading.Lock():
                new_memory = deque([], maxlen=new_capacity)
                if new_capacity > self.capacity and new_capacity >= self.__len__():
                    new_memory.extend(self.memory)
                elif new_capacity > self.capacity and new_capacity < self.__len__():
                    new_memory.extend(self.memory[:self.__len__()])
                elif new_capacity < self.capacity and new_capacity <= self.__len__():
                    new_memory.extend(self.memory[self.__len__() - new_capacity:])
                elif new_capacity < self.capacity and new_capacity > self.__len__():
                    new_memory.extend(self.memory)
                self.memory.clear()
                self.memory = new_memory
                self.capacity = new_capacity

    def add(self, *args):
        """ Save a transition (state, action, next_state, reward) """
        with threading.Lock():
            self.memory.append(Transition(*args))

    def push(self, *args):
        """ Save a transition (state, action, next_state, reward) """
        self.add(*args)

    def sample(self, batch_size):
        with threading.Lock():
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get(self):
        return self.memory


class MpReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def buffer_resize(self, new_capacity: int):
        if self.capacity != new_capacity:
            with MpLock():
                new_memory = deque([], maxlen=new_capacity)
                if new_capacity > self.capacity and new_capacity >= self.__len__():
                    new_memory.extend(self.memory)
                elif new_capacity > self.capacity and new_capacity < self.__len__():
                    new_memory.extend(self.memory[:self.__len__()])
                elif new_capacity < self.capacity and new_capacity <= self.__len__():
                    new_memory.extend(self.memory[self.__len__() - new_capacity:])
                elif new_capacity < self.capacity and new_capacity > self.__len__():
                    new_memory.extend(self.memory)
                self.memory.clear()
                self.memory = new_memory
                self.capacity = new_capacity

    def __len__(self):
        return len(self.memory)

    def add(self, *args):
        """ Save a transition (state, action, next_state, reward) """
        with MpLock():
            self.memory.append(Transition(*args))

    def push(self, *args):
        """ Save a transition (state, action, next_state, reward) """
        self.add(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get(self):
        return self.memory


class MpReplayBufferManager(SyncManager):
    def __init__(self, init_start=True, *args, **kwargs):
        SyncManager.__init__(self, *args, **kwargs)
        self.register('MpReplayBuffer', MpReplayBuffer, exposed=['__len__', 'push', 'add',
                                                                 'sample', 'get', 'buffer_resize'])
        if init_start:
            self.start()


if __name__ == '__main__':
    pass
    #     # rb = ThReplayBuffer(capacity=3)
    #     # rb.push([1, 2, 3, 4], 2, [2, 3, 4, 5], -1)
    #     # print(list(rb.memory))
    #     # experience = rb.sample(1)
    #     # batch = Transition(*zip(*experience))
    #     # print(batch.next_state)

    #     mprb = MpReplayBuffer(capacity=3)
    #     mprb.push([1, 2, 3, 4], 2, [2, 3, 4, 5], -1)
    #     print(list(mprb.memory))
    #     experience = mprb.sample(1)
    #     batch = Transition(*zip(*experience))
    #     print(batch.next_state)
