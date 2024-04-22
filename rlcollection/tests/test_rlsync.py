import unittest
from multiprocessing import Process, Value, Array
from rlcollection.rlsync import RlSync, RLSYNC_obj


class TestRlSync(unittest.TestCase):
    def setUp(self):
        # self.rlsync = RlSync()
        self.rlsync = RLSYNC_obj
        self.n = 100
        self.num_processes = 200

    def test_add_time_step(self):
        def increment_time_step(rlsync):
            for _ in range(self.n):
                rlsync.add_time_step()

        with self.rlsync:
            processes = []
            for i in range(self.num_processes):
                p = Process(target=increment_time_step, args=(self.rlsync,))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            self.assertEqual(self.rlsync.get_time_step(), self.n * self.num_processes)

    def test_set_total_time_steps(self):
        def set_total_time_steps(rlsync, value):
            rlsync.set_total_time_steps(value)

        with self.rlsync:
            processes = []
            for i in range(self.num_processes):
                p = Process(target=set_total_time_steps, args=(self.rlsync, self.n))
                processes.append(p)
                p.start()
                p.join()

            self.assertEqual(self.rlsync.get_total_time_steps(), self.n)

    def test_append_episodes_rewards(self):
        def append_episodes_rewards(rlsync, value):
            for ix in range(value):
                rlsync.append_episodes_rewards(ix)

        with self.rlsync:
            processes = []

            for i in range(self.num_processes):
                p = Process(target=append_episodes_rewards, args=(self.rlsync, self.n))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            self.assertEqual(len(self.rlsync.episodes_rewards), self.num_processes * self.n)
            expected_sorted_values = [val for sublist in zip(*([range(self.n)] * self.num_processes)) for val in
                                      sublist]
            self.assertEqual(sorted(self.rlsync.episodes_rewards), expected_sorted_values)

    def test_append_episodes_length(self):
        def append_episodes_length(rlsync, value):
            for ix in range(value):
                rlsync.append_episodes_length(ix)

        with self.rlsync:
            processes = []

            for i in range(self.num_processes):
                p = Process(target=append_episodes_length, args=(self.rlsync, self.n))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            self.assertEqual(len(self.rlsync.episodes_length), self.num_processes * self.n)
            expected_sorted_values = [val for sublist in zip(*([range(self.n)] * self.num_processes)) for val in
                                      sublist]
            self.assertEqual(sorted(self.rlsync.episodes_length), expected_sorted_values)

    def test_get(self):
        num_processes = 12
        n = 100
        processes = []

        def add_transition(rlsync):
            for i in range(n):
                rlsync.memory.add(i, i + 1, i + 2, i + 3)

        with self.rlsync:
            for _ in range(num_processes):
                p = Process(target=add_transition, args=(self.rlsync,))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            sorted_values = sorted(self.rlsync.memory.get(), key=lambda x: x.state)
            sorted_values = [transition.state for transition in sorted_values]
            expected_sorted_values = [val for sublist in zip(*([range(n)] * num_processes)) for val in sublist]
            self.assertEqual(expected_sorted_values, sorted_values)



if __name__ == '__main__':
    unittest.main()
