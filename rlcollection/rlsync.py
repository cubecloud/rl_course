import logging
from rlcollection.rlmutex import rlmutex
from rlcollection.replaybuffer import ReplayBuffer

__version__ = 0.013

logger = logging.getLogger()


class SingletonClass(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SingletonClass, cls).__new__(cls)
        return cls.instance


class RlSync(SingletonClass):
    __agents_running: int = 0
    __total_time_steps: int = 0
    __time_step: int = 0
    __total_episodes: int = 0
    __episode_num: int = 0
    __net_weights: dict = {}
    __episodes_rewards: list = []
    __episodes_length: list = []
    memory = ReplayBuffer()
    lock = rlmutex

    def reset(cls):
        cls.__agents_running = 0
        cls.__total_time_steps = 0
        cls.__time_step = 0
        cls.__total_episodes: int = 0
        cls.__episode_num: int = 0
        cls.__net_weights: dict = {}
        cls.__episodes_rewards: list = []
        cls.__episodes_length: list = []
        cls.memory.clear()

    def add_agents_running(cls):
        if cls.__agents_running > 0:
            with cls.lock:
                cls.__agents_running += 1
        else:
            cls.__agents_running += 1

    def get_agents_running(cls):
        return cls.__agents_running

    def add_time_step(cls):
        with cls.lock:
            cls.__time_step += 1

    def add_time_steps(cls, value):
        with cls.lock:
            cls.__time_step += value

    def get_time_step(cls):
        return cls.__time_step

    def add_episode_num(cls):
        with cls.lock:
            cls.__episode_num += 1

    def get_episode_num(cls):
        return cls.__episode_num

    def set_total_episodes(cls, value):
        if cls.__agents_running > 1:
            with cls.lock:
                cls.__total_episodes = value
        else:
            cls.__total_episodes = value

    def get_total_episodes(cls):
        return cls.__total_episodes

    def set_total_time_steps(cls, value):
        if cls.__agents_running > 1:
            with cls.lock:
                cls.__total_time_steps = value
        else:
            cls.__total_time_steps = value

    def get_total_time_steps(cls):
        return cls.__total_time_steps

    def append_episodes_rewards(cls, value, id_num):
        with cls.lock:
            # logger.debug(f'append_episodes_rewards {id_num} setting the lock')
            cls.__episodes_rewards.append(value)
        # logger.debug(f'append_episodes_rewards {id_num} release the lock')

    def append_episodes_length(cls, value, id_num):
        with cls.lock:
            # logger.debug(f'append_episodes_length {id_num} setting the lock')
            cls.__episodes_length.append(value)
        # logger.debug(f'append_episodes_length {id_num} release the lock')

    def get_episodes_rewards(cls):
        return cls.__episodes_rewards

    def get_episodes_length(cls):
        return cls.__episodes_length

    def get_weights(cls, agent_id):
        return cls.__net_weights[agent_id]

    def save_weights(cls, agent_id, weights):
        if cls.__agents_running > 1:
            with cls.lock:
                cls.__net_weights.update({agent_id: weights})
        else:
            cls.__net_weights.update({agent_id: weights})


RLSYNC_obj = RlSync()

if __name__ == '__main__':
    print(id(RLSYNC_obj))
    new_sync_obj = RlSync()
    print(id(new_sync_obj))
