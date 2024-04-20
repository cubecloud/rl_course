import random
import threading
from collections import namedtuple, deque

threads_lock = threading.Lock()


__version__ = 0.001

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def buffer_resize(self, new_capacity: int):
        if self.capacity != new_capacity:
            with threads_lock:
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
        self.memory.append(Transition(*args))

    def push(self, *args):
        """ Save a transition (state, action, next_state, reward) """
        self.add(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    rb = ReplayBuffer(capacity=3)
    rb.push([1, 2, 3, 4], 2, [2, 3, 4, 5], -1)
    print(list(rb.memory))
    experience = rb.sample(1)
    batch = Transition(*zip(*experience))
    print(batch.next_state)