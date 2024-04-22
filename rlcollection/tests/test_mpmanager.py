from multiprocessing import Process
from multiprocessing.managers import SyncManager

n = 100
num_processes = 100


class CustomManager(SyncManager):
    def __init__(self, *args):
        super().__init__()
        self.start()
        self.a_int = self.Value('i', 0)
        self.a_list = self.list()
        self.lock = self.Lock()


def custom_set_var(cm,):
    for x in range(n):
        with cm.lock:
            cm.a_int.value += 1
        with cm.lock:
            cm.a_list.append(x)


if __name__ == "__main__":
    with CustomManager() as c_manager:
        processes = [Process(target=custom_set_var, args=(c_manager,)) for _ in range(num_processes)]
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print(c_manager.a_int.value)

        print(sorted(c_manager.a_list))
        print(len(c_manager.a_list))
