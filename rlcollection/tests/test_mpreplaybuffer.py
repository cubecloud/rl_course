import unittest
from rlcollection.replaybuffer import MpReplayBufferManager
from multiprocessing import Process


class TestMpReplayBuffer(unittest.TestCase):

    def test_multiprocessing(self):
        rpm_obj = MpReplayBufferManager()
        buffer = rpm_obj.MpReplayBuffer()
        num_processes = 12
        n = 100
        processes = []

        def add_transition(buffer):
            for i in range(n):
                buffer.add(i, i+1, i+2, i+3)

        with rpm_obj:
            for _ in range(num_processes):
                p = Process(target=add_transition, args=(buffer,))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
            self.assertEqual(len(buffer), n*num_processes)

    def test_get(self):
        rpm_obj = MpReplayBufferManager()
        buffer = rpm_obj.MpReplayBuffer()
        num_processes = 12
        n = 100
        processes = []

        def add_transition(buffer):
            for i in range(n):
                buffer.add(i, i+1, i+2, i+3)

        with rpm_obj:
            for _ in range(num_processes):
                p = Process(target=add_transition, args=(buffer,))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            sorted_values = sorted(buffer.get(), key=lambda x: x.state)
            sorted_values = [transition.state for transition in sorted_values]
            expected_sorted_values = [val for sublist in zip(*([range(n)] * num_processes)) for val in sublist]
            self.assertEqual(expected_sorted_values, sorted_values)


if __name__ == '__main__':
    unittest.main()
