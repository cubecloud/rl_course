import threading
from rlcollection.replaybuffer import ReplayBuffer

__version__ = 0.002

threads_lock = threading.Lock()


class Syncing:
    sync_total_time_steps: int = 0
    sync_time_step: int = 0
    memory = ReplayBuffer()
    lock: threading.Lock = threads_lock


SYNC_obj = Syncing()
