import sys
import time
import logging

import numpy as np
import gymnasium as gym

from threading import Thread

from abc import abstractmethod
from typing import List, Union

from tqdm import tqdm

from rlcollection.rlsync import RLSYNC_obj
from rlcollection.rlagents import DQNAgent

import matplotlib.pyplot as plt
import pandas as pd

__version__ = 0.013

logger = logging.getLogger()


class RLBase:
    def __init__(self, env, agent=DQNAgent, agents_num=1, agents_devices: list = ('cpu',), seed: int = 42, **kwargs):
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
        self.agent = agent
        self.agents_num: int = agents_num
        self.agents_devices: List[str,] = agents_devices
        self.agents_base: list = []
        self.seed: int = seed
        self.net = None
        self.episodes_rewards: list = []
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

    def agents_init(self):
        for ix, device in zip(range(self.agents_num), self.agents_devices):
            self.agents_base.append(self.agent(self.env, seed=self.seed, device=device))

    def save_movie_gif(self, gif_path_filename: str, weights_path_filename: Union[str, None] = None):
        if not self.agents_base:
            self.agents_init()

        if weights_path_filename is not None:
            self.agents_base[0].load(weights_path_filename)
        episode_reward, episode_length = self.agents_base[0].save_movie_gif(gif_path_filename)
        print('Episode length:', episode_length)
        print('Episode reward:', episode_reward)

    def reset(self):
        RLSYNC_obj.reset()
        for ix in range(self.agents_num):
            self.agents_base[ix].reset()

    def learning_curve(self, return_results=False) -> pd.DataFrame:
        window_size = max(int(len(self.episodes_rewards) / 140), 50)
        results_df = pd.DataFrame(data={'scores': self.episodes_rewards, 'frames': self.episodes_length})
        results_df[f'scores_MA{window_size}'] = results_df['scores'].rolling(window=window_size).mean()
        results_df[f'frames_MA{window_size}'] = results_df['frames'].rolling(window=window_size).mean()
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, sharey=False)
        results_df[['scores', f'scores_MA{window_size}']].plot(ax=axes[0])
        axes[0].legend()
        axes[0].set_ylabel("Score")
        axes[0].set_title('Episode score')
        axes[0].set_xlabel('Episode #')
        axes[0].set_ylim(-300, max(self.episodes_rewards) + 20)

        results_df[['frames', f'frames_MA{window_size}']].plot(ax=axes[1])
        axes[1].legend()
        axes[1].set_ylabel("Frames")
        axes[1].set_title('Episode frames length')
        axes[1].set_xlabel('Episode #')
        axes[1].set_ylim(0, max(self.episodes_length) + 20)
        plt.show()
        if return_results:
            return results_df


class RLDQN(RLBase):
    def __init__(self, env, agent=DQNAgent, agents_num=1, agents_devices: list = ('cpu',), seed: int = 42, **kwargs):
        """
        Reinforcement learning DQN algorithm realisation with multithreading and shared replay buffer
        Args:
            env:                        environment object
            agents_num (int):           agents number
            agents_devices List[str,]:  agents devices
            seed(int):                  random seed
            *kwargs:
        """
        super().__init__(env, agent, agents_num, agents_devices, seed, **kwargs)

        if len(agents_devices) != agents_num:
            logger.exception(f"{self.__class__.__name__} Exception: Q-ty of agents_num and agents_devices must equal")
            sys.exit(1)

    def learn(self, total_timesteps):
        self.agents_init()
        self.total_timesteps = total_timesteps

        if RLSYNC_obj.sync_total_time_steps == 0 or RLSYNC_obj.sync_total_time_steps != total_timesteps:
            RLSYNC_obj.sync_total_time_steps = total_timesteps

        def pbar_updater(rlsync_obj):
            pbar = tqdm(total=int(rlsync_obj.sync_total_time_steps))
            while rlsync_obj.sync_time_step <= rlsync_obj.sync_total_time_steps:
                pbar.n = rlsync_obj.sync_time_step
                pbar.refresh()
                time.sleep(.7)
            pbar.n = RLSYNC_obj.sync_time_step
            pbar.refresh()
            pbar.close()

        def agent_thread(idx: int, rlsync_obj):
            while rlsync_obj.sync_time_step <= rlsync_obj.sync_total_time_steps:
                try:
                    episode_reward, episode_length = self.agents_base[idx].episode_learn()
                    with rlsync_obj.lock:
                        self.episodes_rewards.append(episode_reward)
                        self.episodes_length.append(episode_length)
                except (KeyboardInterrupt, SystemExit):
                    pass

        threads_lst: list = []
        for ix in range(len(self.agents_base)):
            threads_lst.append(Thread(target=agent_thread, args=(ix, RLSYNC_obj), name=f'DQN_{ix}'))
            threads_lst[-1].start()
            with RLSYNC_obj.lock:
                RLSYNC_obj.agents_running += 1

        threads_lst.append(Thread(target=pbar_updater, args=(RLSYNC_obj,), name=f'pbar'))
        threads_lst[-1].start()
        for thread in threads_lst:
            thread.join()


if __name__ == '__main__':
    frames_to_learn = 50_000
    env = gym.make("LunarLander-v2",
                   render_mode=None,
                   continuous=False,
                   gravity=-9.8,
                   enable_wind=True,
                   wind_power=15.0,
                   turbulence_power=1.5,
                   )

    rl = RLDQN(env, DQNAgent,  agents_num=4, agents_devices=['cuda', 'cuda', 'cuda', 'cpu'])
    rl.learn(frames_to_learn)
    rl.save(f'./simpledqqn_{frames_to_learn}.pth')
    rl.learning_curve()
