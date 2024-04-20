import logging

import numpy as np
import gymnasium as gym

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Thread

from abc import abstractmethod
from typing import List, Union

from tqdm import tqdm

from rlcollection.rlsync import SYNC_obj
from rlcollection.rlagents import DQNAgent

import matplotlib.pyplot as plt

__version__ = 0.007

logger = logging.getLogger()


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
        self.total_rewards: list = []
        self.episodes_length: list = []
        self._check_params(kwargs)
        self.total_timesteps = 0

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
        self.agents_base: list = []
        assert len(agents_devices) == agents_num, "Error: Q-ty of agents_num and agents_devices must equal"
        for ix, device in zip(range(self.agents_num), self.agents_devices):
            self.agents_base.append(DQNAgent(env, seed=self.seed, device=device))

    def learn(self, total_timesteps):
        self.total_timesteps = total_timesteps
        threads: list = []
        if SYNC_obj.sync_total_time_steps == 0 or SYNC_obj.sync_total_time_steps != total_timesteps:
            SYNC_obj.sync_total_time_steps = total_timesteps

        def thread_target(idx: int):
            try:
                episode_reward, episode_length = self.agents_base[idx].agent_learn()
                with SYNC_obj.lock:
                    pbar.update(episode_length)
                    self.total_rewards.append(episode_reward)
                    self.episodes_length.append(episode_length)
            except (KeyboardInterrupt, SystemExit):
                pass

        pbar = tqdm(total=SYNC_obj.sync_total_time_steps)
        while SYNC_obj.sync_time_step < SYNC_obj.sync_total_time_steps:
            threads: list = []
            for ix in range(len(self.agents_base)):
                threads.append(Thread(target=thread_target,
                                      args=(ix,),
                                      name=f'DQN_{ix}')
                               )
                threads[-1].start()
                threads[-1].join()
        pbar.refresh()
        pbar.close()

    def save_movie_gif(self, gif_path_filename: str, weights_path_filename: Union[str, None] = None):
        if weights_path_filename is not None:
            self.agents_base[0].load(weights_path_filename)
        episode_reward, episode_length = self.agents_base[0].save_movie_gif(gif_path_filename)
        print('Episode length:', episode_length)
        print('Episode reward:', episode_reward)

    def learning_curve(self):
        pic_length = 15
        window_size = max(int(len(self.total_rewards)/10_000), 5)
        moving_avg_rewards = np.convolve(self.total_rewards, np.ones(window_size) / window_size, mode='valid')
        moving_avg_lengths = np.convolve(self.episodes_length, np.ones(window_size) / window_size, mode='valid')

        plt.figure(figsize=(pic_length, 10))
        plt.title('Rewards')
        plt.plot(moving_avg_rewards)
        plt.show()
        plt.figure(figsize=(pic_length, 10))
        plt.title('Frames length')
        plt.plot(moving_avg_lengths)
        plt.show()


if __name__ == '__main__':
    frames_to_learn = 500_000
    env = gym.make("LunarLander-v2",
                   render_mode=None,
                   continuous=False,
                   gravity=-9.8,
                   enable_wind=True,
                   wind_power=15.0,
                   turbulence_power=1.5,
                   )

    rl = RLDQN(env, agents_num=1, agents_devices=['cuda', ])
    rl.learn(frames_to_learn)
    rl.save(f'./simpledqqn_{frames_to_learn}.pth')
