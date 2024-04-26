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

__version__ = 0.015

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
        self.agents_devices: List[str,] = self.prepare_agents_devices(agents_devices)
        self.agents_base: list = []
        self.seed: int = seed
        self.net = None
        self.episodes_rewards: list = []
        self.episodes_length: list = []
        self._check_params(kwargs)
        self.total_timesteps: int = 0
        self.total_episodes: int = 0

    def _check_params(self, params):
        for k, v in params.items():
            if not hasattr(self, k):
                logger.warning(f"{self.__class__.__name__} is missing parameter '{k}'")
            setattr(self, k, v)
        return params

    def prepare_agents_devices(self, agents_devices):
        if len(agents_devices) != self.agents_num:
            logger.exception(f"{self.__class__.__name__} Exception: Q-ty of agents_num and agents_devices must equal")
            sys.exit(1)
        _agents_devices = []
        for k, v in dict(zip(*np.unique(agents_devices, return_counts=True))).items():
            for i in range(v):
                _agents_devices.append(f'{k}:{i}')
        return _agents_devices

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

    def set_env(self, env):
        self.env = env
        for agent in self.agents_base:
            agent.set_env(self.env)

    def agents_init(self):
        for ix, device in zip(range(self.agents_num), self.agents_devices):
            self.agents_base.append(self.agent(self.env, seed=self.seed, device=device))

    def save_movie_gif(self, gif_path_filename: str, weights_path_filename: Union[str, None] = None):
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
        fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True, sharey=False)
        results_df[['scores', f'scores_MA{window_size}']].plot(ax=axes[0])
        axes[0].legend()
        axes[0].set_ylabel("Score")
        axes[0].set_title('Episode score')
        axes[0].set_xlabel('Episode #')
        axes[0].set_ylim(-300, max(self.episodes_rewards) + 100)

        results_df[['frames', f'frames_MA{window_size}']].plot(ax=axes[1])
        axes[1].legend()
        axes[1].set_ylabel("Frames")
        axes[1].set_title('Episode frames length')
        axes[1].set_xlabel('Episode #')
        axes[1].set_ylim(0, max(self.episodes_length) + 100)
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

    def learn(self, total_condition, condition='time_step', progress_bar=True):
        self.agents_init()

        if condition == 'time_step':
            self.total_timesteps = total_condition
            constraint = lambda p: p.get_time_step() <= p.get_total_time_steps()
            if RLSYNC_obj.get_total_time_steps() == 0 or RLSYNC_obj.get_total_time_steps() != total_condition:
                RLSYNC_obj.set_total_time_steps(total_condition)
        elif condition == 'episode':
            self.total_episodes = total_condition
            constraint = lambda p: p.get_episode_num() <= p.get_total_episodes()
            if RLSYNC_obj.get_total_episodes() == 0 or RLSYNC_obj.get_total_episodes() != total_condition:
                RLSYNC_obj.set_total_episodes(total_condition)
        else:
            assert condition in ['time_step', 'episode'], f'Error: unknown condition {condition}'

        def pbar_updater(rlsync_obj, constraint):
            pbar = tqdm(total=total_condition)
            if condition == 'time_step':
                func = rlsync_obj.get_time_step
                sl_time = 0.7
            else:
                func = rlsync_obj.get_episode_num
                sl_time = 1.2
            while constraint(rlsync_obj):
                pbar.n = func()
                pbar.refresh()
                time.sleep(sl_time)
            pbar.n = func()
            pbar.refresh()
            pbar.close()

        def agent_thread(idx: int, rlsync_obj, ):
            try:
                self.agents_base[idx].agent_learn(rlsync_obj=rlsync_obj, constraint=constraint, condition=condition)
            except (KeyboardInterrupt, SystemExit):
                pass

        threads_lst: list = []
        for ix in range(len(self.agents_base)):
            threads_lst.append(Thread(target=agent_thread, args=(ix, RLSYNC_obj), name=f'DQN_{ix}'))
            threads_lst[-1].start()
            # RLSYNC_obj.add_agents_running()

        if progress_bar:
            threads_lst.append(Thread(target=pbar_updater, args=(RLSYNC_obj, constraint), name=f'pbar'))
            threads_lst[-1].start()
        for thread in threads_lst:
            thread.join()

        self.episodes_rewards = RLSYNC_obj.get_episodes_rewards()
        self.episodes_length = RLSYNC_obj.get_episodes_length()


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('check_lock.log')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL.Image').setLevel(logging.WARNING)
    """----------------------------------------------------------------"""

    to_learn = 500
    env = gym.make("LunarLander-v2",
                   render_mode=None,
                   continuous=False,
                   gravity=-9.8,
                   enable_wind=True,
                   wind_power=15.0,
                   turbulence_power=1.5,
                   )

    # rl = RLDQN(env, DQNAgent, agents_num=3, agents_devices=['cuda', 'cpu', 'cpu'])
    rl = RLDQN(env, DQNAgent, agents_num=2, agents_devices=['cpu', 'cpu'])
    rl.learn(500, condition='episode', progress_bar=True)
    rl.save(f'./test_{to_learn}.pth')
    """ Testing Load """
    env = gym.make("LunarLander-v2",
                   render_mode='rgb_array_list',
                   continuous=False,
                   gravity=-9.8,
                   enable_wind=True,
                   wind_power=15.0,
                   turbulence_power=1.5,
                   )
    rl.set_env(env)
    gif_path_filename = f'./test_{to_learn}.gif'
    rl.save_movie_gif(gif_path_filename, f'./test_{to_learn}.pth')
    # rl.learning_curve()
