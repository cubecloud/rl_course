import os
import sys
import time
import logging
import datetime
from pytz import timezone
import numpy as np
import gymnasium as gym

from threading import Thread

from typing import List, Union

from tqdm import tqdm

# from rlcollection.configdqn_lunarlanderv2 import ConfigAgent
# from rlcollection.configa2c_lunarlanderv2 import ConfigAgent
# from rlcollection.configa2c_cartpolev1 import ConfigAgent
from deeprl.rlsync import RLSYNC_obj
from deeprl.rlagents import DQNAgent
from deeprl.rlagents import A2CAgent
from deeprl.rlagents import show_mp4, show_records

import matplotlib.pyplot as plt
import pandas as pd

__version__ = 0.025

TZ = timezone('Europe/Moscow')

logger = logging.getLogger()


class RLBase:
    algorithm_name = None

    def __init__(self, env, agent=DQNAgent, agents_num=1, config=None, agents_devices: list = ('cpu',),
                 seed: int = 42, **kwargs):
        """
        Reinforcement learning realization with multithreading and shared replay buffer
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
        self.ConfigAgent = config
        self.set_exp_id()
        self.create_exp_dirs()

    def create_exp_dirs(self):
        dirs = dict()
        dirs['exp'] = os.path.join(self.ConfigAgent.EXPERIMENT_PATH,
                                   self.ConfigAgent.ENV_NAME,
                                   self.algorithm_name,
                                   self.ConfigAgent.EXP_ID)
        dirs['training'] = os.path.join(dirs['exp'], 'training')
        dirs['evaluation'] = os.path.join(dirs['exp'], 'evaluation')
        os.makedirs(dirs['training'], exist_ok=True)
        os.makedirs(dirs['evaluation'], exist_ok=True)
        setattr(self.ConfigAgent, 'ALGO', self.algorithm_name)
        setattr(self.ConfigAgent, 'DIRS', dirs)

    def set_exp_id(self):
        setattr(self.ConfigAgent, 'EXP_ID', f'exp-{datetime.datetime.now(TZ).strftime("%y%m%d-%H%M%S")}')

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
        elif len(agents_devices) == 1:
            return agents_devices
        _agents_devices = []
        for k, v in dict(zip(*np.unique(agents_devices, return_counts=True))).items():
            for i in range(v):
                if k == 'cuda:':
                    _agents_devices.append(f'{k}:{i}')
                else:
                    _agents_devices.append(f'{k}')
        return _agents_devices

    def set_net(self, network_obj):
        self.net = network_obj

    def save(self, pathname: str):
        self.agents_base[0].save(pathname)

    def load(self, pathname, weights_only):
        for agent in self.agents_base:
            agent.load(pathname, weights_only)

    def get_env(self):
        return self.env

    def set_env(self, env):
        self.env = env
        for agent in self.agents_base:
            agent.set_env(self.env)

    def agents_init(self):
        for ix, device in zip(range(self.agents_num), self.agents_devices):
            self.agents_base.append(self.agent(self.env, seed=self.seed, device=device, config=self.ConfigAgent))

    def save_movie_gif(self, gif_path_filename: str, weights_path_filename: Union[str, None] = None):
        self.agents_init()

        if weights_path_filename is not None:
            self.agents_base[0].load(weights_path_filename)

        episode_reward, episode_length = self.agents_base[0].save_movie_gif(gif_path_filename)
        print('Episode length:', episode_length)
        print('Episode reward:', episode_reward)

    def save_movie_mp4(self, mp4_path_filename: str, weights_path_filename: Union[str, None] = None):
        self.agents_init()

        if weights_path_filename is not None:
            self.agents_base[0].load(weights_path_filename)

        episode_reward, episode_length = self.agents_base[0].save_movie_mp4(mp4_path_filename)
        print('Episode length:', episode_length)
        print('Episode reward:', episode_reward)

    def show_mp4(self, mp4_path_filename: str):
        return show_mp4(mp4_path_filename)

    def reset(self):
        RLSYNC_obj.reset()
        for ix in range(self.agents_num):
            self.agents_base[ix].reset()

    def learning_curve(self, return_results=False, save_figure=True, show_figure=False) -> pd.DataFrame:
        window_size = max(int(len(self.episodes_rewards) / 140), 100)
        results_df = pd.DataFrame(data={'scores': self.episodes_rewards, 'frames': self.episodes_length})
        results_df[f'scores_MA{window_size}'] = results_df['scores'].rolling(window=window_size).mean()
        results_df[f'frames_MA{window_size}'] = results_df['frames'].rolling(window=window_size).mean()
        fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True, sharey=False)
        results_df[['scores', f'scores_MA{window_size}']].plot(ax=axes[0])
        axes[0].legend()
        axes[0].set_ylabel("Score")
        axes[0].set_title('Episode score')
        axes[0].set_xlabel('Episode #')
        rewards_range = abs(max(self.episodes_rewards) - min(self.episodes_rewards))
        axes[0].set_ylim(min(self.episodes_rewards) - (rewards_range * 0.1),
                         max(self.episodes_rewards) + (rewards_range * 0.1))

        results_df[['frames', f'frames_MA{window_size}']].plot(ax=axes[1])
        axes[1].legend()
        axes[1].set_ylabel("Frames")
        axes[1].set_title('Episode frames length')
        axes[1].set_xlabel('Episode #')
        axes[1].set_ylim(0, max(self.episodes_length) * 1.15)
        path_filename = os.path.join(self.ConfigAgent.DIRS['training'], 'learning_curve.svg')
        if save_figure:
            plt.savefig(path_filename,
                        dpi=450,
                        facecolor='w',
                        edgecolor='w',
                        orientation='portrait',
                        format=None,
                        transparent=False,
                        bbox_inches=None,
                        pad_inches=0.1,
                        metadata=None
                        )
        if show_figure:
            plt.show()
        if return_results:
            return results_df

    def fit(self, total_condition, condition='time_step', progress_bar=True, use_checkpoint_dir=None,
            weights_only=False):
        self.agents_init()
        if use_checkpoint_dir is not None:
            self.load(use_checkpoint_dir, weights_only)

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
                val_metric = rlsync_obj.get_val_metrics()
                last_win_share: float = 0.
                last_avg_step_count: float = 0.
                last_avg_reward: float = 0.
                last_val_step: int = 0
                if val_metric:
                    last_val_step = max(val_metric.keys())
                    last_win_share = val_metric[last_val_step]['win_ratio']
                    last_avg_reward = val_metric[last_val_step]['avg_reward']
                    last_avg_step_count = val_metric[last_val_step]['avg_step_count']
                msg = (
                    f"Frame: {rlsync_obj.get_time_step():09d} | "
                    f"Terminated/truncated: {rlsync_obj.get_terminated()}/{rlsync_obj.get_truncated()} | "
                    f"Epsilon: {rlsync_obj.get_eps_threshold():.4f} | "
                    f"Val {last_val_step:06d}: win_ratio={last_win_share:.2f}, avg_reward={last_avg_reward:.2f}, "
                    f"avg_steps: {last_avg_step_count:.1f}")
                pbar.set_description(msg)
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
            threads_lst.append(Thread(target=agent_thread, args=(ix, RLSYNC_obj), name=f'{self.algorithm_name}_{ix}'))
            threads_lst[-1].start()
            # RLSYNC_obj.add_agents_running()

        if progress_bar:
            threads_lst.append(Thread(target=pbar_updater, args=(RLSYNC_obj, constraint), name=f'pbar'))
            threads_lst[-1].start()
        for thread in threads_lst:
            thread.join()

        self.episodes_rewards = RLSYNC_obj.get_episodes_rewards()
        self.episodes_length = RLSYNC_obj.get_episodes_length()

    def evaluate(self, episodes=3, reward_condition=None, use_checkpoint_dir=None, weights_only=False):
        if not self.agents_base:
            self.agents_init()
            if use_checkpoint_dir is not None:
                self.load(use_checkpoint_dir, weights_only)
        self.agents_base[0].evaluation(episodes, reward_condition)
        pass


class RLDQN(RLBase):
    algorithm_name = 'DQN'

    def __init__(self, env, agent=DQNAgent, agents_num=1, config=None, agents_devices: list = ('cpu',), seed: int = 42,
                 **kwargs):
        """
        Reinforcement learning DQN algorithm realisation with multithreading and shared replay buffer
        Args:
            env:                        environment object
            agents_num (int):           agents number
            agents_devices List[str,]:  agents devices
            seed(int):                  random seed
            *kwargs:
        """
        super().__init__(env, agent, agents_num, config, agents_devices, seed, **kwargs)


class RLA2C(RLBase):
    algorithm_name = 'A2C'

    def __init__(self, env, agent=A2CAgent, agents_num=1, agents_devices: list = ('cpu',), seed: int = 42,
                 config=None, **kwargs):
        """
        Reinforcement learning A2C algorithm realization with multithreading and shared replay buffer
        Args:
            env:                        environment object
            agents_num (int):           agents number
            agents_devices List[str,]:  agents devices
            seed(int):                  random seed
            *kwargs:
        """
        super().__init__(env, agent, agents_num, agents_devices, seed, config, **kwargs)


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
    """ Testing for Lunarlander """
    from rlcollection.configa2c_lunarlanderv2 import ConfigAgent

    to_learn = 60000
    env = gym.make("LunarLander-v2",
                   render_mode='rgb_array',
                   continuous=False,
                   gravity=-9.8,
                   enable_wind=True,
                   wind_power=15.0,
                   turbulence_power=1.5,
                   )
    # env = gym.make("CartPole-v1",
    #                render_mode='rgb_array',
    #                )

    rl = RLDQN(env, DQNAgent, agents_num=3, agents_devices=['cuda', 'cpu', 'cpu'], config=ConfigAgent)
    # rl = RLDQN(env, DQNAgent, agents_num=2, agents_devices=['cuda', 'cpu', ], config=ConfigAgent)
    # rl = RLDQN(env, DQNAgent, agents_num=1, agents_devices=['cpu'], config=ConfigAgent)
    # rl = RLA2C(env, A2CAgent, agents_num=1, agents_devices=['cuda'], config=ConfigAgent)
    # rl = RLA2C(env, A2CAgent, agents_num=1, agents_devices=['cuda'], config=ConfigAgent)
    # rl = RLA2C(env, A2CAgent, agents_num=2, agents_devices=['cuda', 'cpu'], config=ConfigAgent)
    rl.fit(to_learn,
           condition='episode',
           progress_bar=True,
           use_checkpoint_dir='/home/cubecloud/Python/projects/rl_course/rlcollection/LunarLander-v2/A2C/exp-1306-150004/training/eps-34000',
           weights_only=False,
           )
    rl.evaluate(5)
    rl.learning_curve(show_figure=True)
