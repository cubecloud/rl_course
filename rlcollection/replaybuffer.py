import random
from rlcollection.rlmutex import rlmutex
from collections import namedtuple, deque

__version__ = 0.004

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def buffer_resize(self, new_capacity: int):
        if self.capacity != new_capacity:
            with rlmutex:
                new_memory = deque([], maxlen=new_capacity)
                if new_capacity > self.capacity or (self.capacity > new_capacity > self.__len__()):
                    new_memory.extend(self.memory)
                elif new_capacity < self.capacity and new_capacity <= self.__len__():
                    new_memory.extend(self.memory[self.__len__() - new_capacity:])
                self.memory.clear()
                self.memory = new_memory
                self.capacity = new_capacity

    def add(self, *args):
        """ Save a transition (state, action, next_state, reward) """
        with rlmutex:
            self.memory.append(Transition(*args))

    def push(self, *args):
        """ Save a transition (state, action, next_state, reward) """
        self.add(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def extend(self, batch_list):
        with rlmutex:
            self.memory.extend(batch_list)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        with rlmutex:
            self.memory.clear()


if __name__ == '__main__':
    rb = ReplayBuffer(capacity=3)
    rb.push([1, 2, 3, 4], 2, [2, 3, 4, 5], -1)
    print(list(rb.memory))
    experience = rb.sample(1)
    batch = Transition(*zip(*experience))
    print(batch.next_state)
