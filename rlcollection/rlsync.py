from rlcollection.rlmutex import threads_lock
from rlcollection.replaybuffer import ReplayBuffer

__version__ = 0.011


class SingletonClass(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SingletonClass, cls).__new__(cls)
        return cls.instance


class RlSync(SingletonClass):
    agents_running = 0
    sync_total_time_steps: int = 0
    sync_time_step: int = 0
    memory = ReplayBuffer()
    lock = threads_lock
    net_weights: dict = {}
    episodes_rewards: list = []
    episodes_length: list = []

    @classmethod
    def get_weights(cls, agent_id):
        return cls.net_weights[agent_id]

    @classmethod
    def save_weights(cls, agent_id, weights):
        cls.net_weights.update({agent_id: weights})

    @classmethod
    def reset(cls):
        cls.agents_running = 0
        cls.sync_total_time_steps: int = 0
        cls.sync_time_step: int = 0
        cls.memory = ReplayBuffer()
        cls.net_weights: dict = {}


RLSYNC_obj = RlSync()

if __name__ == '__main__':
    print(id(RLSYNC_obj))
    new_sync_obj = RlSync()
    print(id(new_sync_obj))
