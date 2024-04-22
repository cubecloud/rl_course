import time
import logging

import numpy as np
import gymnasium as gym

from abc import abstractmethod
from typing import List, Union

from tqdm import tqdm

# from rlcollection.rlsync import RlSync, RLSYNC_obj
from rlcollection.rlsync import RlSync

from rlcollection.rlagents import DQNAgent

from multiprocessing import Process
import multiprocessing as mp

import matplotlib.pyplot as plt

__version__ = 0.010

logger = logging.getLogger()


# RLSYNC_obj = RlSync()


class RLBase:
    def __init__(self, env, agents_num=1, agents_devices: list = ('cpu',), seed: int = 42, **kwargs):
        """
        Reinforcement learning DQN algorithm realisation with multithreading and shared replay buffer
        Args:
            env:                        environment object
            agents_num (int):           agents number
            agents_devices List[str,]:  agents devices
            seed(int):                  random seed
            *kwargs:
        """

        self.env = env
        self.agents_num: int = agents_num
        self.agents_devices: List[str,] = agents_devices
        self.agents_base: list = []
        self.seed: int = seed
        self.net = None
        self._check_params(kwargs)
        self.total_timesteps = 0
        # self.RLSYNC_obj = RlSync()

    def _check_params(self, params):
        for k, v in params.items():
            if not hasattr(self, k):
                logger.warning(f"{self.__class__.__name__} is missing parameter '{k}'")
            setattr(self, k, v)
        return params

    def set_net(self, network_obj):
        self.net = network_obj

    def save(self, path_filename: str):
        self.agents_base[0].save(path_filename)

    def load(self, path_filename):
        for agent in self.agents_base:
            agent.load(path_filename)

    def get_env(self):
        return self.env

    @abstractmethod
    def learn(self, total_timesteps: int):
        pass

    @abstractmethod
    def save_movie_gif(self, gif_path_filename: str, weights_path_filename: Union[str, None] = None):
        pass


class RLDQN(RLBase):
    def __init__(self, env, agents_num=1, agents_devices: list = ('cpu',), seed: int = 42, **kwargs):
        """
        Reinforcement learning DQN algorithm realisation with multithreading and shared replay buffer
        Args:
            env:                        environment object
            agents_num (int):           agents number
            agents_devices List[str,]:  agents devices
            seed(int):                  random seed
            *kwargs:
        """
        super().__init__(env, agents_num, agents_devices, seed, **kwargs)
        assert len(agents_devices) == agents_num, "Error: Q-ty of agents_num and agents_devices must equal"
        self.agents_devices = agents_devices
        self.agents_num = agents_num

    def pbar_updater(self, rlsync_obj, total):

        pbar = tqdm(total=total)
        while rlsync_obj.get_time_step() <= rlsync_obj.get_total_time_steps():
            pbar.n = rlsync_obj.get_time_step()
            pbar.refresh()
        pbar.close()

    def learn(self, total_timesteps):
        self.total_timesteps = total_timesteps

        for ix, device in zip(range(self.agents_num), self.agents_devices):
            self.agents_base.append(DQNAgent(env, seed=self.seed, device=device))

        RLSYNC_obj = RlSync()

        with RLSYNC_obj:
            if RLSYNC_obj.get_total_time_steps() == 0 or RLSYNC_obj.get_total_time_steps() != total_timesteps:
                RLSYNC_obj.set_total_time_steps(total_timesteps)

            #   setup agents base

            # pbar = tqdm(total=RLSYNC_obj.get_total_time_steps())
            mp_pool: list = []

            # mp_pool.append(Process(target=self.pbar_updater, args=(RLSYNC_obj, RLSYNC_obj.get_total_time_steps()),
            #                        name=f'pbar_{ix}'))
            # mp_pool[-1].start()

            for ix, agent in enumerate(self.agents_base):
                mp_pool.append(
                    Process(target=self.agents_base[ix].agent_learn, args=(RLSYNC_obj,), name=f'DQN_{ix + 1}'))
                # mp_pool[ix].daemon = True
                mp_pool[ix].start()

            # self.pbar_updater(RLSYNC_obj)
            # while RLSYNC_obj.get_time_step() <= RLSYNC_obj.get_total_time_steps():
            #     pbar.n = RLSYNC_obj.get_time_step()
            #     pbar.refresh()
            # pbar.close()
            # for ix in range(len(mp_pool)):
            #     mp_pool[ix].join()
            # for ix in range(0, len(self.agents_base)):
            #     self.agents_base[ix].stop_running = True
            # for ix in range(1, len(self.agents_base)):
            #     self.agents_base[ix].stop_running = True
            #     mp_pool[ix].join()
            # if mp_pool[ix].is_alive():
            #     mp_pool[ix].terminate()
            # mp_pool[ix].terminate()

    def save_movie_gif(self, gif_path_filename: str, weights_path_filename: Union[str, None] = None):
        if weights_path_filename is not None:
            self.agents_base[0].load(weights_path_filename)
        episode_reward, episode_length = self.agents_base[0].save_movie_gif(gif_path_filename)
        print('Episode length:', episode_length)
        print('Episode reward:', episode_reward)

    def save(self, path_filename: str):
        _temp_agent = DQNAgent(env, seed=self.seed, device=self.agents_devices[0])
        _temp_agent.setup_agent(RLSYNC_obj)
        _temp_agent.get_weights_from_rlsync()
        _temp_agent.save(path_filename)

    def learning_curve(self):
        pic_length = 15
        window_size = max(int(len(RLSYNC_obj.episodes_rewards) / 10_000), 5)
        moving_avg_rewards = np.convolve(RLSYNC_obj.episodes_rewards, np.ones(window_size) / window_size,
                                         mode='valid')
        moving_avg_lengths = np.convolve(RLSYNC_obj.episodes_length, np.ones(window_size) / window_size,
                                         mode='valid')

        plt.figure(figsize=(pic_length, 10))
        plt.title('Rewards')
        plt.plot(moving_avg_rewards)
        plt.show()
        plt.figure(figsize=(pic_length, 10))
        plt.title('Frames length')
        plt.plot(moving_avg_lengths)
        plt.show()


if __name__ == '__main__':
    # mp.set_start_method("spawn")
    frames_to_learn = 3_000
    env = gym.make("LunarLander-v2",
                   render_mode=None,
                   continuous=False,
                   gravity=-9.8,
                   enable_wind=True,
                   wind_power=15.0,
                   turbulence_power=1.5,
                   )

    rl = RLDQN(env, agents_num=2, agents_devices=['cpu', 'cpu'])
    rl.learn(frames_to_learn)
    # rl.save(f'./simpledqqn_{frames_to_learn}.pth')
