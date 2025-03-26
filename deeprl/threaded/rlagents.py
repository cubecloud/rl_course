import os
import io

import copy
import math
import json
import random

import numpy as np
import pandas as pd

from itertools import count

from abc import ABC, abstractmethod

import torch
import torch.optim as optim
import torch.nn.functional as F

from typing import Tuple, Union, Dict, List, Callable, Optional

from deeprl.threaded.rlnetworks import FCnet, EMBEDnet, ActorNet, ContinuousActorNet
from deeprl.threaded.replaybuffer import Transition, PPOTransition
from deeprl.threaded.rlsync import RLSYNC_obj
from deeprl.threaded.rltools import saveanimation, save_mp4, play_and_save_mp4
from deeprl.threaded.rltools import RewardsNormalizer
from deeprl.threaded.rltools import EnvFrameStackedWrapper
from deeprl.threaded.rltools import scale_car_actions

from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import TransformObservation
import gymnasium as gym

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

__version__ = 0.079


class AgentMeta(ABC):
    id_count: int = 0
    agent_algo = 'BASE'

    def __init__(self, env_kwargs, seed, config, device='cpu', net_model=FCnet):
        """
        Args:
            env_kwargs:     environment kwargs
            seed (int):     random seed
        """
        self.env_kwargs = env_kwargs
        self.env = gym.make(**env_kwargs)
        self.id_num = int(self.id_count)
        self.ConfigAgent = config
        self.device = device
        self.n_actions = None
        self.state_size: Union[int, None] = None
        self.net_model = net_model
        self.seed: int = seed + self.id_num
        self.condition = None
        self.history = None
        self.val_agent = None
        self.use_env_wrapper = False

    def get_frames_sequence(self, render=True):
        frames = []
        # terminated = 0
        state, info = self.env.reset()
        episode_reward = 0.
        done = False
        while not done:
            action = self.get_action(state, info)
            next_state, reward, terminated, truncated, info = self.step(state, action)
            episode_reward += reward
            if render:
                frames.append(self.env.render())
            else:
                frames.append(1)
            done = terminated or truncated
            state = next_state
        return frames, episode_reward, done

    def evaluation(self, evaluation_eps_count, reward_condition):
        if reward_condition is None:
            reward_condition = self.ConfigAgent.REWARD_CONDITION
        new_env_kwargs = copy.deepcopy(self.env_kwargs)
        new_env_kwargs.update({'render_mode': 'rgb_array'})
        if self.use_env_wrapper and isinstance(self.env, EnvFrameStackedWrapper):
            old_initial_kwargs = copy.copy(self.env.initial_kwargs.copy())
            new_initial_kwargs = copy.copy(old_initial_kwargs)
            new_initial_kwargs['env_kwargs'] = new_env_kwargs
            new_initial_kwargs['env'] = None
            self.env = self.env.__class__(**new_initial_kwargs)
        else:
            self.env = gym.make(**new_env_kwargs)

        for eps_num in range(1, evaluation_eps_count + 1):
            eps_frames, eps_reward, eps_final_state = self.get_frames_sequence()
            path_filename = os.path.join(self.ConfigAgent.DIRS["evaluation"],
                                         f'{self.ConfigAgent.ALGO}-{self.ConfigAgent.EXP_ID}-eps-{eps_num:02d}.mp4')

            save_mp4(eps_frames, path_filename)
            print(f'EPISODE # {eps_num}')
            if eps_final_state == 1 and eps_reward >= reward_condition:
                print('Agent reached the Goal 🏆')
            else:
                print('Agent loose ☠️')
            print(f'Reward: {eps_reward}')
            print(f'Number of steps: {len(eps_frames)}')
            print('-' * 50)

        if self.use_env_wrapper and isinstance(self.env, EnvFrameStackedWrapper):
            self.env = self.env.__class__(**old_initial_kwargs)
        else:
            self.env = gym.make(**self.env_kwargs)

    def validation(self, validation_eps_num, validation_eps_count, ):
        won_eps_count = 0
        current_train_dir = os.path.join(self.ConfigAgent.DIRS["training"], f'eps-{validation_eps_num}')
        os.makedirs(current_train_dir, exist_ok=True)
        val_eps_to_check = int(self.ConfigAgent.VALIDATION_STEP * self.ConfigAgent.VALIDATION_RATIO)
        eps_step_counts, eps_reward_counts, eps_done_counts = self.validate_episodes(num_episodes=val_eps_to_check)
        for eps_reward, eps_done in zip(eps_reward_counts, eps_done_counts):
            if eps_done and eps_reward >= self.ConfigAgent.REWARD_CONDITION:
                won_eps_count += 1

        old_env = self.env
        new_env_kwargs = copy.deepcopy(self.env_kwargs)
        new_env_kwargs.update({'render_mode': 'rgb_array'})

        if self.use_env_wrapper and isinstance(self.env, EnvFrameStackedWrapper):
            # old_initial_kwargs = copy.copy(self.env.initial_kwargs.copy())
            new_initial_kwargs = copy.copy(self.env.initial_kwargs.copy())
            new_initial_kwargs['env_kwargs'] = new_env_kwargs
            new_initial_kwargs['env'] = None
            self.env = self.env.__class__(**new_initial_kwargs)
        else:
            self.env = gym.make(**new_env_kwargs)

        for eps_num in range(1, validation_eps_count + 1):
            eps_frames, eps_reward, eps_done = self.get_frames_sequence()
            mp4_path_filename = os.path.join(current_train_dir,
                                             f'{self.ConfigAgent.ALGO}-{self.ConfigAgent.EXP_ID}-eps-{eps_num:02d}.mp4')
            save_mp4(eps_frames, mp4_path_filename)

        # return old_env with 'render_mode = None'
        self.env = old_env
        # if self.use_env_wrapper and isinstance(self.env, EnvFrameStackedWrapper):
        #     self.env = self.env.__class__(**old_initial_kwargs)
        # else:
        #     self.env = gym.make(**self.env_kwargs)
        val_metrics: dict = {'episode': validation_eps_num,
                             'win_ratio': round(won_eps_count / val_eps_to_check, 3),
                             'avg_reward': round(sum(eps_reward_counts) / val_eps_to_check, 1),
                             'avg_step_count': round(sum(eps_step_counts) / val_eps_to_check, 2),
                             'epsilon': round(self.RLSYNC_obj.get_eps_threshold(), 3)
                             }

        self.RLSYNC_obj.set_val_metrics({validation_eps_num: val_metrics})
        self.save_metrics(val_metrics, 'validation', current_train_dir)
        self.save(pathname=current_train_dir, checkpoint_num=validation_eps_num)
        history_path_filename = os.path.join(self.ConfigAgent.DIRS["training"], 'history.csv')
        if self.history is None:
            self.history = pd.DataFrame(data=val_metrics, index=[validation_eps_num])
        else:
            self.history = pd.concat([self.history, pd.DataFrame(data=val_metrics, index=[validation_eps_num])],
                                     ignore_index=True)
        self.save_history(history_path_filename)

    def validate_episodes(self, num_episodes: int = 3, truncate_steps: int = 1000):
        episodes_steps, episodes_rewards, episodes_dones = [], [], []
        for episodes in range(num_episodes):
            eps_steps, eps_reward, eps_done = self.validate_episode(truncate_steps=truncate_steps)
            episodes_steps.append(eps_steps)
            episodes_rewards.append(eps_reward)
            episodes_dones.append(eps_done)
        return episodes_steps, episodes_rewards, episodes_dones

    def validate_episode(self, truncate_steps: int = 1000):
        eps_steps, eps_reward = 0, 0.
        eps_done = False
        state, info = self.env.reset()
        for eps_steps in range(truncate_steps):
            action = self.get_action(state, info)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            eps_reward += reward
            state = next_state
            eps_done = terminated or truncated
            if eps_done:
                break
        return eps_steps, eps_reward, eps_done

    def agent_learn(self, rlsync_obj=None, constraint=None, condition='time_step'):
        if rlsync_obj is None:
            self.RLSYNC_obj = RLSYNC_obj
        else:
            self.RLSYNC_obj = rlsync_obj

        self.condition = condition
        self.RLSYNC_obj.add_agents_running()
        if condition == "time_step":
            val_steps = list(range(0, RLSYNC_obj.get_total_time_steps() + 1, self.ConfigAgent.VALIDATION_STEP)[1:])
        else:
            val_steps = list(range(0, RLSYNC_obj.get_total_episodes() + 1, self.ConfigAgent.VALIDATION_STEP)[1:])

        def is_validation_step(validation_steps):
            if condition == "time_step":
                if self.RLSYNC_obj.get_time_step() >= validation_steps[0]:
                    self.validation(validation_steps[0], self.ConfigAgent.VALIDATION_EPS)
                    return True
            else:
                if self.RLSYNC_obj.get_episode_num() >= validation_steps[0]:
                    self.validation(validation_steps[0], self.ConfigAgent.VALIDATION_EPS)
                    return True
            return False

        while constraint(self.RLSYNC_obj):
            episode_reward, episode_length = self.episode_learn()
            self.RLSYNC_obj.append_episodes_rewards(episode_reward, self.id_num)
            self.RLSYNC_obj.append_episodes_length(episode_length, self.id_num)
            self.RLSYNC_obj.add_episode_num()
            if self.id_num == 0:
                if val_steps:
                    if is_validation_step(val_steps):
                        val_steps = val_steps[1:]

    def select_action(self, state, info, eps_threshold: Union[float or None] = None):
        """
        Returns action for discrete action space with epsilon greedy action selection to explore space

        Args:

            state (np.array):               state (observations) as np.array
            info (dict):                    can contain action mask
            eps_threshold (float or None):  eps_threshold for validating agent
        Returns:
            action:     action
        """
        if eps_threshold is None:
            eps_threshold = self.ConfigAgent.EPS_END + (
                    self.ConfigAgent.EPS_START - self.ConfigAgent.EPS_END) * math.exp(
                -1. * self.RLSYNC_obj.get_time_step() / self.ConfigAgent.EPS_DECAY)

        self.RLSYNC_obj.set_eps_threshold(eps_threshold)

        if random.random() > eps_threshold:
            return self.get_action(state, info)
        else:
            if self.ConfigAgent.MASKED_ACTION and "action_mask" in info.keys():
                return self.env.action_space.sample(info["action_mask"])
            else:
                return self.env.action_space.sample()

    def get_action(self, state, info):
        """
        Returns random action from action space for BASE method
        Args:
            state:  dummy arg for base class
            info:   dummy arg for base class, used for action mask
        Returns:
            action:     action
        """
        return self.env.action_space.sample()

    @abstractmethod
    def transform_reward(self, state, observation, reward, terminated, truncated, info):
        """
        Function wrapper for transforming rewards for experiments - empty wrapper
        Args:
            state:
            observation:
            reward:
            terminated:
            truncated:
            info:

        Returns:
            reward:     transformed reward score
        """
        return reward

    def step(self, current_state, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    @abstractmethod
    def episode_learn(self) -> Tuple[float, int]:
        pass

    @abstractmethod
    def set_env(self, env):
        pass

    def save_metrics(self, metrics, metrics_type, train_eps_dir):
        metrics_path = os.path.join(train_eps_dir, f'{metrics_type}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

    def save_history(self, path_filename):
        self.history.to_csv(path_filename)

    def prepared_for_device(self, state_dict):
        """
        Used to transfer weights from one device to another to same network architecture
        Args:
            state_dict (dict):  weights dictionary

        Returns:
            weights (dict):
        """
        temp_buffer = io.BytesIO()
        torch.save(state_dict, temp_buffer)
        temp_buffer.seek(0)
        return torch.load(temp_buffer, map_location=self.device)

    @abstractmethod
    def save(self, pathname: str, checkpoint_num: int):
        pass

    @abstractmethod
    def load(self, pathname: str):
        pass


class DQNAgent(AgentMeta):
    """Interacts with and learns from environment."""
    id_count: int = -1
    agent_algo = 'DQN'

    def __init__(self, env_kwargs, seed, config, device='cpu', net_model=FCnet, filters_base_size: int = 96,
                 states_queue_size: int = 1):
        """
        Args:
            env_kwargs:     environment kwargs
            seed (int):     random seed
        """
        super().__init__(env_kwargs, seed, config, device, net_model)
        self.filters_base_size = 128
        self.embed_in = 500
        self.embed_out = 4
        random.seed(self.seed)
        self.states_queue_size = states_queue_size
        self.filters_base_size = filters_base_size
        self.states_dtype = torch.float32
        self.policy_net_kwargs: dict = {}
        self.target_net_kwargs: dict = {}
        self.policy_net = None
        self.target_net = None
        self.set_env(self.env)

        self.memory = None
        self.optimizer = None

        # self.l1_cache_size = self.ConfigAgent.BATCH_SIZE * 4
        self.l1_cache: list = []
        self.local_timestep: int = 0

        self.__RLSYNC_obj = None
        self.update_id_count()
        self.id_num = int(self.id_count)

        if self.ConfigAgent.MASKED_ACTION:
            self.action_funct = self._get_masked_action
        else:
            self.action_funct = self._get_unmasked_action

    def set_env(self, env):
        self.env = env
        self.env.reset()
        action_sample = self.env.action_space.sample()
        step_sample = self.env.step(action_sample)
        self.n_actions = self.env.action_space.n
        if isinstance(self.env.observation_space, Discrete):
            """ wrapper for discrete space with size = 1 """
            self.states_dtype = torch.long
            self.net_model = EMBEDnet
            self.env = TransformObservation(self.env, lambda state: np.array([state]))
            if isinstance(step_sample[0], (int or float)):
                self.state_size = self.states_queue_size
                self.embed_in = self.env.observation_space.n
                self.embed_out = 2 * self.n_actions
            else:
                self.state_size = len(step_sample[0]) * self.states_queue_size
                self.embed_out = 4
            self.policy_net_kwargs = {'vocab_size': self.embed_in,
                                      'embed_out': self.embed_out,
                                      'context_size': self.states_queue_size,
                                      'n_actions': self.n_actions,
                                      'l1_filters': 16 * self.states_queue_size,
                                      'seed': self.seed}
            self.target_net_kwargs = self.policy_net_kwargs
        elif isinstance(self.env.observation_space, Box):
            # self.env = TransformObservation(self.env, lambda state: np.asarray(state))
            self.state_size = len(self.env.observation_space.high)
            self.policy_net_kwargs = {'state_size': self.state_size * self.states_queue_size,
                                      'n_actions': self.n_actions,
                                      'l1_filters': int(self.states_queue_size * self.filters_base_size),
                                      'l2_filters': int(self.states_queue_size * self.filters_base_size),
                                      'seed': self.seed
                                      }
            self.target_net_kwargs = self.policy_net_kwargs

    @classmethod
    def update_id_count(cls):
        cls.id_count += 1

    @property
    def RLSYNC_obj(self):
        return self.__RLSYNC_obj

    @RLSYNC_obj.setter
    def RLSYNC_obj(self, obj):
        self.__RLSYNC_obj = obj
        #   setting shared mp or multithreaded variables through RLSYNC_obj
        self.memory = self.__RLSYNC_obj.memory
        self.memory.buffer_resize(self.ConfigAgent.BUFFER_SIZE)

        self.net_init(self.net_model, self.net_model, self.policy_net_kwargs, self.target_net_kwargs)

    def net_init(self, policy_net, target_net, policy_net_kwargs, target_net_kwargs):
        with self.__RLSYNC_obj.lock:
            self.policy_net = policy_net(**policy_net_kwargs).to(self.device)
            self.target_net = target_net(**target_net_kwargs).to(self.device)

            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.ConfigAgent.LR, amsgrad=True)
            self.save_transfer_weights()

    def select_action(self, state, info, eps_threshold: Union[float or None] = None):
        """
        Returns action for discrete action space with epsilon greedy action selection to explore space

        Args:

            state (np.array):               state (observations) as np.array
            info (dict):                    can contain action mask
            eps_threshold (float or None):  eps_threshold for validating agent
        Returns:
            action:     action
        """

        eps_threshold = self.RLSYNC_obj.get_eps_threshold()

        if random.random() > eps_threshold:
            return self.get_action(state, info)
        else:
            if self.ConfigAgent.MASKED_ACTION and "action_mask" in info.keys():
                return self.env.action_space.sample(info["action_mask"])
            else:
                return self.env.action_space.sample()

    def _get_unmasked_action(self, state, info):
        q_values = self.policy_net(state).cpu().squeeze().numpy()
        action = q_values.argmax(axis=-1)
        return action

    def _get_masked_action(self, state, info):
        mask = ~np.array(info["action_mask"], dtype=bool)
        q_values = self.policy_net(state).cpu().squeeze().numpy()
        action = np.ma.masked_array(q_values, mask=mask, fill_value=-np.inf).argmax(axis=0)
        # action = np.argmax(q_values[state, np.where(info["action_mask"] == 1)[0]])
        return action

    def get_action(self, state, info):
        """
        Returns action for discrete action space from learned network

        Args:
            info (dict):
            state (np.array):   state (observations) as np.array
        Returns:
            action:     action
        """
        with torch.no_grad():
            state = torch.tensor(state, dtype=self.states_dtype, device=self.device).unsqueeze(0)
            # t.max(1) вернет наибольшее значение столбца в каждой строке.
            # Второй столбец в результате max - это индекс того места,
            # где был найден максимальный элемент,
            # поэтому мы выбираем действие с наибольшим ожидаемым вознаграждением.
            action = self.action_funct(state, info)
            return action
            # return self.policy_net(state).max(1)[1].view(1, 1).item()

    def optimize_model(self, batch_size):
        """
        Args:
            batch_size(int):    batch_size
        """

        def calc_gradient_algo(state_batch, action_batch, reward_batch):
            # Вычислить маску нефинальных состояний и соединить элементы батча
            # (финальным состоянием должно быть то, после которого моделирование закончилось)
            t_non_final_mask = torch.tensor(_non_final_mask, device=self.device,
                                            dtype=torch.bool)

            t_non_final_next_states = torch.tensor(_non_final_next_states, device=self.device)

            # Собираем батчи для состояний, действий и наград
            # state_batch = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
            state_batch = torch.tensor(state_batch, dtype=self.states_dtype, device=self.device)
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)

            # Вычислить Q(s_t, a) - модель вычисляет Q(s_t),
            # затем мы выбираем столбцы предпринятых действий.
            # Это те действия, которые были бы предприняты для каждого состояния партии в соответствии с policy_net
            # state_batch.shape = (256, 1) or (256, x)
            # action_batch.shape = (256, 1)
            state_action_values = self.policy_net(state_batch).gather(-1, action_batch)
            # Вычислить V(s_{t+1}) для всех следующих состояний.
            # Ожидаемые значения действий для не_финальных_следующих_состояний вычисляются
            # на основе "старшей" целевой_сети; выбирается их наилучшее вознаграждение с помощью max(1)[0].
            # Это объединяется по маске, так что мы будем иметь либо ожидаемое значение состояния,
            # либо 0, если состояние было финальным.
            next_state_values = torch.zeros(batch_size, device=self.device)
            with torch.no_grad():
                next_state_values[t_non_final_mask] = self.target_net(t_non_final_next_states).max(1)[0]
            # Вычисляем ожидаемые Q значения
            expected_state_action_values = (next_state_values * self.ConfigAgent.GAMMA) + reward_batch

            # Объединяем все в общий лосс
            # criterion = torch.nn.HuberLoss(delta=2.)
            criterion = torch.nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Готовим градиент
            self.optimizer.zero_grad()
            loss.backward()
            # Обрезаем значения градиента - проблемма исчезающего/взрывающего градиента
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)

            self.optimizer.step()

        if RLSYNC_obj.get_agents_running() > 1:
            with RLSYNC_obj.lock:
                experience = self.memory.sample(batch_size)
                batch = Transition(*zip(*experience))
                _non_final_mask = np.array(tuple(map(lambda s: s is not None, batch.next_state)))
                _non_final_next_states = np.array([s for s in batch.next_state if s is not None])
                calc_gradient_algo(np.array(batch.state), np.array(batch.action), np.array(batch.reward))
        else:
            experience = self.memory.sample(batch_size)
            batch = Transition(*zip(*experience))
            _non_final_mask = np.array(tuple(map(lambda s: s is not None, batch.next_state)))
            _non_final_next_states = np.array([s for s in batch.next_state if s is not None])
            calc_gradient_algo(np.array(batch.state), np.array(batch.action), np.array(batch.reward))

        # update target net if necessary
        self.soft_update()

    def target_net_update(self):
        policy_net_state_dict = self.policy_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.ConfigAgent.TAU + target_net_state_dict[
                key] * (1 - self.ConfigAgent.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def agents_net_updates(self):
        def state_dict_update(local_dict, other_dict):
            for key in other_dict:
                local_dict[key] = other_dict[key] * a_TAU + local_dict[key] * (1 - a_TAU)
            return local_dict

        a_TAU = self.ConfigAgent.TAU
        agents_lst = list(range(self.RLSYNC_obj.get_agents_running()))
        agents_lst.remove(self.id_num)

        local_policy_net_state_dict = self.policy_net.state_dict()
        local_target_net_state_dict = self.target_net.state_dict()

        for agent_id in agents_lst:
            self.load_transfer_weights(agent_id)
            other_policy_net_state_dict = self.policy_net.state_dict()
            other_target_net_state_dict = self.target_net.state_dict()
            self.policy_net.load_state_dict(state_dict_update(local_policy_net_state_dict, other_policy_net_state_dict))
            self.target_net.load_state_dict(state_dict_update(local_target_net_state_dict, other_target_net_state_dict))

    def soft_update(self):
        """  Soft update model parameters """
        if self.local_timestep % self.ConfigAgent.SYNC == 0:
            # θ′ ← τ θ + (1 −τ )θ′
            self.target_net_update()
        if self.RLSYNC_obj.get_agents_running() > 1:
            if self.local_timestep % self.ConfigAgent.AGENTS_SYNC == 0:
                self.save_transfer_weights()
                # θ′ ← τ θ + (1 −τ )θ′
                self.agents_net_updates()

    @abstractmethod
    def transform_reward(self, state, observation, reward, terminated, truncated, info):
        """
        Function wrapper for transforming rewards for experiments - empty wrapper
        Args:
            state:
            observation:
            reward:
            terminated:
            truncated:
            info:

        Returns:
            reward:     transformed reward score
        """
        return reward

    def step(self, current_state, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # reward = self.transform_reward(current_state, observation, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info

    def episode_learn(self) -> Tuple[float, int]:
        episode_reward = 0
        # Для каждого эпизода инициализируем начальное состояние
        state, info = self.env.reset()
        # frame_idx - считает сколько шагов успели сделать пока не закончился эпизод
        for frame_idx in count():

            action = self.select_action(state, info, eps_threshold=None)

            # Делаем шаг агентом
            observation, reward, terminated, truncated, info = self.step(state, action)
            episode_reward += reward

            # Объединяем done по двум конечным состояниям
            done = terminated or truncated
            # присваиваем следующее состояние
            if terminated:
                next_state = None
            else:
                next_state = observation
            # making a step in train
            self.train_step(state, [action], next_state, reward, flush=done)

            # update the timesteps +1
            self.RLSYNC_obj.add_time_step()
            self.local_timestep += 1
            state = next_state
            # переходим на следующее состояние
            if done:
                if truncated:
                    self.RLSYNC_obj.add_truncated()
                elif terminated:
                    self.RLSYNC_obj.add_terminated()
                eps_threshold = self.ConfigAgent.EPS_END + (
                        self.ConfigAgent.EPS_START - self.ConfigAgent.EPS_END) * math.exp(
                    -1. * self.RLSYNC_obj.get_time_step() / self.ConfigAgent.EPS_DECAY)
                self.RLSYNC_obj.set_eps_threshold(eps_threshold)
                break

        return episode_reward, frame_idx + 1

    def train_step(self, state, action, next_state, reward, flush=False):
        # Creating L1 cache to unsync agents with replay buffer
        self.l1_cache.append([state, action, next_state, reward])
        if flush:
            self.memory.extend([Transition(*element) for element in self.l1_cache])
            # print(f'\n{self.ConfigAgent.EPS_DECAY}//{self.RLSYNC_obj.get_time_step()}//{self.eps_threshold}')
            self.l1_cache.clear()

        if self.memory.ready:
            self.optimize_model(self.ConfigAgent.BATCH_SIZE)

    def save_weights(self, path_filename_wo_ext: str):
        torch.save(self.policy_net.state_dict(), f'{path_filename_wo_ext}_policy.pth')
        torch.save(self.target_net.state_dict(), f'{path_filename_wo_ext}_target.pth')

    def load_weights(self, path_filename_wo_ext: str):
        if self.RLSYNC_obj is None:
            self.RLSYNC_obj = RLSYNC_obj

        # __weights = torch.load(path_filename)
        self.policy_net.load_state_dict(self.prepared_for_device(torch.load(f'{path_filename_wo_ext}_policy.pth')))
        self.target_net.load_state_dict(self.prepared_for_device(torch.load(f'{path_filename_wo_ext}_target.pth')))

    def load(self, pathname: str, weights_only=False):
        checkpoint_num = pathname.split('-')[-1]
        path_filename_wo_ext = os.path.join(pathname, f'{self.agent_algo}-{checkpoint_num}')
        self.load_weights(path_filename_wo_ext)
        if not self.memory.ready:
            if not weights_only:
                self.memory.load(f'{path_filename_wo_ext}.pkl')

    def save(self, pathname: str, checkpoint_num: int):
        path_filename_wo_ext = os.path.join(pathname, f'{self.agent_algo}-{checkpoint_num}')
        self.save_weights(f'{path_filename_wo_ext}')
        self.memory.save(f'{path_filename_wo_ext}.pkl')

    def save_transfer_weights(self):
        RLSYNC_obj.save_weights(self.id_num, [copy.deepcopy(self.policy_net.state_dict()),
                                              copy.deepcopy(self.target_net.state_dict())])

    def load_transfer_weights(self, agent_id):
        __weights = self.RLSYNC_obj.get_weights(agent_id)
        self.policy_net.load_state_dict(self.prepared_for_device(__weights[0]))
        self.target_net.load_state_dict(self.prepared_for_device(__weights[1]))

    def reset(self):
        self.l1_cache.clear()
        self.net_init(self.net_model, self.net_model, self.policy_net_kwargs, self.target_net_kwargs)


class A2CAgent(AgentMeta):
    """ Interacts with and learns from environment."""

    id_count: int = -1
    agent_algo = 'A2C'

    def __init__(self, env_kwargs, seed, config, device='cpu', net_model=ActorNet, filters_base_size: int = 96,
                 states_queue_size: int = 1):
        """
        Args:
            env:            initialized environment class
            seed (int):     random seed
        """
        super().__init__(env_kwargs, seed, config, device, net_model)
        self.__RLSYNC_obj = None
        self.update_id_count()
        self.id_num = int(self.id_count)
        self.filters_base_size: int = filters_base_size
        random.seed(self.seed)
        self.states_queue_size = states_queue_size
        self.actor_net_kwargs: dict = {}
        self.value_net_kwargs: dict = {}
        self.actor_net = None
        self.value_net = None
        self.action_type: str = 'discrete'
        self.get_action_method: Optional[Callable] = None
        self.n_epochs: int = 10
        self.set_env(self.env)
        self.rewards_normalizer = RewardsNormalizer(normalizer_method='std')
        self.memory = None
        self.actor_optimizer = None
        self.value_optimizer = None

        self.l1_cache: list = []
        self.local_timestep: int = 0
        self.local_episode: int = 0

    @classmethod
    def update_id_count(cls):
        cls.id_count += 1

    @property
    def RLSYNC_obj(self):
        return self.__RLSYNC_obj

    @RLSYNC_obj.setter
    def RLSYNC_obj(self, obj):
        self.__RLSYNC_obj = obj
        #   setting shared mp or multithreaded variables through RLSYNC_obj
        self.memory = self.__RLSYNC_obj.memory
        self.memory.buffer_resize(self.ConfigAgent.BUFFER_SIZE)
        self.net_init(self.net_model, self.net_model, self.actor_net_kwargs, self.value_net_kwargs)

    def net_init(self, actor_net, value_net, actor_net_kwargs, value_net_kwargs):
        with self.__RLSYNC_obj.lock:
            self.actor_net = actor_net(**actor_net_kwargs).to(self.device)
            self.value_net = value_net(**value_net_kwargs).to(self.device)

            self.actor_optimizer = optim.AdamW(self.actor_net.parameters(), lr=self.ConfigAgent.LR)
            self.value_optimizer = optim.AdamW(self.value_net.parameters(), lr=self.ConfigAgent.LR)
            self.save_transfer_weights()

    def set_env(self, env):
        self.env = copy.deepcopy(env)
        self.env.reset()
        value_out_filters = 1
        actor_last_activation = None
        if isinstance(self.env.action_space, Discrete):
            self.n_actions = self.env.action_space.n
            self.get_action_method = self._get_discrete_action

        if isinstance(self.env.observation_space, Box):
            self.state_size = len(self.env.observation_space.high)
            self.actor_net_kwargs = {'state_size': self.state_size * self.states_queue_size,
                                     'out_filters': self.n_actions,
                                     'l1_filters': int(self.states_queue_size * self.filters_base_size),
                                     'l2_filters': int(self.states_queue_size * self.filters_base_size),
                                     'seed': self.seed,
                                     'last_activation': actor_last_activation,
                                     }
            self.value_net_kwargs = {'state_size': self.state_size * self.states_queue_size,
                                     'out_filters': value_out_filters,
                                     'l1_filters': int(self.states_queue_size * self.filters_base_size),
                                     'l2_filters': int(self.states_queue_size * self.filters_base_size),
                                     'seed': self.seed,
                                     }

    def _get_discrete_action(self, state, info, eps_threshold: Optional[float] = None):
        with torch.no_grad():
            state_batch = np.expand_dims(state, axis=0)
            state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)

            # Get logits from state
            logits = self.actor_net(state_batch).squeeze()

            # From logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Pick up action's sample
            action = torch.multinomial(probs, num_samples=1)
            # tolist move tensor to cpu if necessary
            return action.tolist()[0]

    def get_action(self, state, info, eps_threshold: Optional[float] = None):
        return self.get_action_method(state, info, eps_threshold)

    @abstractmethod
    def transform_reward(self, state, observation, reward, terminated, truncated, info):
        pass

    def episode_learn(self) -> Tuple[float, int]:
        episode_reward = 0
        state, info = self.env.reset()
        # frame_idx - counting steps
        for frame_idx in count():

            action = self.select_action(state, info, eps_threshold=None)

            # step
            observation, reward, terminated, truncated, info = self.step(state, action)
            episode_reward += reward

            done = terminated or truncated
            # next_state = observation -> if NOT terminated
            if terminated:
                next_state = None
            else:
                next_state = observation

            # update the timesteps +1
            self.RLSYNC_obj.add_time_step()
            self.local_timestep += 1

            # training step
            self.train_step(state, [action], next_state, reward, flush=done)

            state = next_state
            if done:
                if truncated:
                    self.RLSYNC_obj.add_truncated()
                elif terminated:
                    self.RLSYNC_obj.add_terminated()
                break
        return episode_reward, frame_idx + 1

    def train_step(self, state, action, next_state, reward, flush=False):
        # Creating L1 cache to unsync agents with replay buffer
        self.l1_cache.append([state, action, next_state, reward])
        if flush:
            self.local_episode += 1
            episode_data = [Transition(*element) for element in self.l1_cache]
            self.optimize_model([Transition(*zip(*episode_data))])
            self.memory.extend_episode(episode_data)
            if RLSYNC_obj.get_agents_running() > 1:
                if self.memory.ready and self.local_episode % self.ConfigAgent.SYNC == 0:
                    episodes_lst = self.memory.sample_episode(self.ConfigAgent.BATCH_SIZE)
                    batch = []
                    for episode in episodes_lst:
                        batch.append(Transition(*zip(*episode)))
                    self.optimize_model(batch)
            self.l1_cache.clear()
            self.soft_update()

    def optimize_model(self, episodes_data):
        for epoch in range(self.n_epochs):
            for episode in episodes_data:
                self._optimize_model_episode(episode)

    def _optimize_model_episode(self, episode_data):
        """
        Args:
            episode_data (named_tuple):    episode_data

        """
        # calculating discounted rewards:
        cum_reward = []
        discounted_reward = 0
        for reward in episode_data.reward[::-1]:
            discounted_reward = reward + self.ConfigAgent.GAMMA * discounted_reward
            cum_reward.append(discounted_reward)

        # create reversed array cos of using append instead insert
        cum_reward = np.array(cum_reward[::-1])

        # using standard normalizer (
        cum_reward = self.rewards_normalizer(cum_reward)

        # setting zero grad for (vf) values function
        self.value_optimizer.zero_grad()

        # creating states tensor
        states = torch.tensor(np.array(episode_data.state), dtype=torch.float32).to(self.device)

        # creating reward (cum_reward) tensor
        cum_reward = torch.tensor(cum_reward, dtype=torch.float32).to(self.device)

        # Calculating loss for vf
        values = self.value_net(states)
        values = values.squeeze(dim=1)
        vf_loss = F.mse_loss(values, cum_reward, reduction="none")

        # values (vf) gradient backpropagation calculation
        vf_loss.sum().backward()

        # making values gradient step
        self.value_optimizer.step()

        # calculating optimized (vf) values
        with torch.no_grad():
            values = self.value_net(states)

        # setting zero grad for (pi) policy function
        self.actor_optimizer.zero_grad()

        # creating actions tensor
        actions = torch.tensor(episode_data.action, dtype=torch.long).to(self.device)
        # calculating advantage function
        # (get (vf) values after gradient update and calculating advantages with real reward)
        advantages = cum_reward - values.squeeze()

        # calculating loss for policy (pi)
        logits = self.actor_net(states)
        log_probs = -F.cross_entropy(logits, actions.squeeze(), reduction="none")
        pi_loss = -log_probs * advantages

        # policy (vf) gradient backpropagation calculation
        pi_loss.sum().backward()

        # making policy gradient step
        self.actor_optimizer.step()

    def agents_net_updates(self):
        def state_dict_update(local_dict, other_dict):
            for key in other_dict:
                local_dict[key] = other_dict[key] * a_TAU + local_dict[key] * (1 - a_TAU)
            return local_dict

        a_TAU = self.ConfigAgent.TAU
        agents_lst = list(range(self.RLSYNC_obj.get_agents_running()))
        agents_lst.remove(self.id_num)

        local_actor_net_state_dict = copy.deepcopy(self.actor_net.state_dict())
        local_value_net_state_dict = copy.deepcopy(self.value_net.state_dict())

        for agent_id in agents_lst:
            self.load_transfer_weights(agent_id)
            other_actor_net_state_dict = copy.deepcopy(self.actor_net.state_dict())
            other_value_net_state_dict = copy.deepcopy(self.value_net.state_dict())
            self.actor_net.load_state_dict(state_dict_update(local_actor_net_state_dict, other_actor_net_state_dict))
            self.value_net.load_state_dict(state_dict_update(local_value_net_state_dict, other_value_net_state_dict))

    def soft_update(self):
        if self.RLSYNC_obj.get_agents_running() > 1:
            self.save_transfer_weights()
            if self.local_episode % self.ConfigAgent.AGENTS_SYNC == 0:
                # θ′ ← τ θ + (1 −τ )θ′
                self.agents_net_updates()

    def load(self, pathname: str, weights_only=False):
        checkpoint_num = pathname.split('-')[-1]
        path_filename_wo_ext = os.path.join(pathname, f'{self.agent_algo}-{checkpoint_num}')
        self.load_weights(path_filename_wo_ext)
        if not self.memory.ready:
            if not weights_only:
                self.memory.load(f'{path_filename_wo_ext}.pkl')
        self.save_transfer_weights()

    def save(self, pathname: str, checkpoint_num: int):
        path_filename_wo_ext = os.path.join(pathname, f'{self.agent_algo}-{checkpoint_num}')
        self.save_weights(f'{path_filename_wo_ext}')
        self.memory.save(f'{path_filename_wo_ext}.pkl')

    def save_weights(self, path_filename_wo_ext: str):
        torch.save(self.actor_net.state_dict(), f'{path_filename_wo_ext}_actor.pth')
        torch.save(self.value_net.state_dict(), f'{path_filename_wo_ext}_value.pth')

    def load_weights(self, path_filename_wo_ext: str):
        if self.RLSYNC_obj is None:
            self.RLSYNC_obj = RLSYNC_obj

        def get_weights(path_filename):
            try:
                __weights = torch.load(path_filename, map_location=self.device)
            except:
                __weights = self.prepared_for_device(torch.load(path_filename))
            return __weights

        self.actor_net.load_state_dict(get_weights(f'{path_filename_wo_ext}_actor.pth'))
        self.value_net.load_state_dict(get_weights(f'{path_filename_wo_ext}_value.pth'))

    def save_transfer_weights(self):
        RLSYNC_obj.save_weights(self.id_num, [copy.deepcopy(self.actor_net.state_dict()),
                                              copy.deepcopy(self.value_net.state_dict())])

    def load_transfer_weights(self, agent_id):
        __weights = self.RLSYNC_obj.get_weights(agent_id)
        with self.RLSYNC_obj.lock:
            try:
                self.actor_net.load_state_dict(__weights[0])
            except:
                self.actor_net.load_state_dict(self.prepared_for_device(__weights[0]))

            try:
                self.value_net.load_state_dict(__weights[1])
            except:
                self.value_net.load_state_dict(self.prepared_for_device(__weights[1]))

    def reset(self):
        self.l1_cache.clear()
        self.net_init(self.actor_net, self.value_net, self.actor_net_kwargs, self.value_net_kwargs)


class PPOAgent(AgentMeta):
    """ Interacts with and learns from environment."""

    id_count: int = -1
    agent_algo = 'PPO'

    def __init__(self, env_kwargs, seed, config, device='cpu', net_model=ActorNet, filters_base_size: int = 96,
                 states_queue_size: int = 1, batch_size: int = 256):
        """
        Args:
            env:            initialized environment class
            seed (int):     random seed
        """
        super().__init__(env_kwargs, seed, config, device, net_model)
        self.__RLSYNC_obj = None
        self.update_id_count()
        self.id_num = int(self.id_count)
        self.filters_base_size: int = filters_base_size
        self.sub_steps_num: int = 0
        # self.fms: Optional[FrameStack] = None
        random.seed(self.seed)
        self.states_queue_size = states_queue_size
        self.actor_net_kwargs: dict = {}
        self.value_net_kwargs: dict = {}
        self.actor_net = None
        self.value_net = None
        self.action_type: str = 'discrete'
        self.discrete: bool = True
        self.get_action_method: Optional[Callable] = None
        self.n_epochs: int = 10
        self.set_env(self.env)
        self.rewards_normalizer = RewardsNormalizer(normalizer_method='std')
        self.memory = None
        self.actor_optimizer = None
        self.value_optimizer = None
        self.batch_size: int = batch_size
        self.l1_cache: list = []
        self.local_timestep: int = 0
        self.local_episode: int = 0
        self.max_grad_norm = 0.5
        self.clip_param = 0.1  # epsilon in clipped loss
        self.ent_coef = 0.2
        self.vf_coef = 1.0
        self.frame_idx = -1

    @classmethod
    def update_id_count(cls):
        cls.id_count += 1

    @property
    def RLSYNC_obj(self):
        return self.__RLSYNC_obj

    @RLSYNC_obj.setter
    def RLSYNC_obj(self, obj):
        self.__RLSYNC_obj = obj
        #   setting shared mp or multithreaded variables through RLSYNC_obj
        self.memory = self.__RLSYNC_obj.memory
        self.memory.buffer_resize(self.ConfigAgent.BUFFER_SIZE)
        self.net_init(self.net_model, self.net_model, self.actor_net_kwargs, self.value_net_kwargs)

    def set_env(self, env):

        self.env = copy.deepcopy(env)
        self.env.reset()
        value_out_filters = 1
        actor_last_activation = None
        self.n_actions = len(self.env.action_space.high)
        if isinstance(self.env.action_space, Discrete):
            self.get_action_method = self._get_discrete_action
            self.action_type = 'discrete'
            self.discrete = True
            if isinstance(self.env.observation_space, Box):
                self.state_size = len(self.env.observation_space.high)
                self.actor_net_kwargs = {'state_size': self.state_size * self.states_queue_size,
                                         'out_filters': self.n_actions,
                                         'l1_filters': int(self.states_queue_size * self.filters_base_size),
                                         'l2_filters': int(self.states_queue_size * self.filters_base_size),
                                         'seed': self.seed,
                                         'last_activation': actor_last_activation,
                                         }
                self.value_net_kwargs = {'state_size': self.state_size * self.states_queue_size,
                                         'out_filters': value_out_filters,
                                         'l1_filters': int(self.states_queue_size * self.filters_base_size),
                                         'l2_filters': int(self.states_queue_size * self.filters_base_size),
                                         'seed': self.seed,
                                         }
        elif isinstance(self.env.action_space, Box):
            self.get_action_method = self._get_continuous_action
            self.action_type = 'continuous'
            self.discrete = False
            if isinstance(self.env.observation_space, Box):
                self.net_model = ContinuousActorNet
                self.filters_base_size = 8
                if 'CarRacing' in self.env_kwargs["id"]:
                    self.use_env_wrapper = True
                    self.env = EnvFrameStackedWrapper(env=self.env,
                                                      env_kwargs=self.env_kwargs,
                                                      stack_frames=4,
                                                      action_repeat=6,
                                                      reward_norm_window=100,
                                                      ma_reward_condition=-0.1,
                                                      color_control=185,
                                                      scale_action=scale_car_actions)
                    # self.sub_steps_num = 4
                    # self.fms = FrameStack(self.sub_steps_num)
                    self.state_size = self.env.fms.num_frames
                else:
                    self.state_size = len(self.env.observation_space.high)

                self.actor_net_kwargs = {'seed': self.seed,
                                         'input_shape': (self.state_size,
                                                         self.env.observation_space.shape[0],
                                                         self.env.observation_space.shape[1]),
                                         'l1_filters': int(self.states_queue_size * self.filters_base_size),
                                         'action_dim': self.n_actions,
                                         }
                self.value_net_kwargs = {}

    def net_init(self, actor_net, value_net, actor_net_kwargs, value_net_kwargs):
        with self.__RLSYNC_obj.lock:
            self.actor_net = actor_net(**actor_net_kwargs).to(self.device)
            self.actor_optimizer = optim.AdamW(self.actor_net.parameters(), lr=self.ConfigAgent.LR)
            if self.discrete:
                self.value_net = value_net(**value_net_kwargs).to(self.device)
                self.value_optimizer = optim.AdamW(self.value_net.parameters(), lr=self.ConfigAgent.LR)
            else:
                self.value_net = self.actor_net
                self.value_optimizer = self.actor_optimizer
            self.save_transfer_weights()

    def _get_discrete_action(self,
                             state: Union[np.ndarray, torch.Tensor],
                             info: Optional[dict] = None,
                             eps_threshold: Optional[float] = None) -> tuple:
        state = torch.from_numpy(state).float().unsqueeze(0)
        logits = self.actor_net(state)
        cat_distribution = torch.distributions.Categorical(logits=logits)
        action = cat_distribution.sample()
        a_logprob = cat_distribution.log_prob(action).sum(dim=1)
        action = action.squeeze().cpu().numpy()
        a_logprob = a_logprob.item()
        return action, a_logprob

    def _get_continuous_action(self,
                               state: Union[np.ndarray, torch.Tensor],
                               info: Optional[dict] = None,
                               eps_threshold: Optional[float] = None) -> tuple:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, sigma = self.actor_net(state)[0]

        normal_distribution = torch.distributions.Normal(mu, sigma)
        action = normal_distribution.sample()
        a_logprob = normal_distribution.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logprob = a_logprob.item()
        return action, a_logprob

    def get_action(self, state, info, eps_threshold: Optional[float] = None):
        if self.use_env_wrapper:
            not_scaled_action, not_scaled_a_logprob = self.get_action_method(state, info, eps_threshold)
            return self.env.scale_action(not_scaled_action)
        return self.get_action_method(state, info, eps_threshold)[0]

    def select_action(self,
                      state: Union[np.ndarray, torch.Tensor],
                      info: Optional[dict] = None,
                      eps_threshold: Union[float or None] = None):
        return self.get_action_method(state, info, eps_threshold)

    def step(self, current_state, action):
        if self.use_env_wrapper:
            self.frame_idx += self.env.action_repeat
        state, reward, terminated, truncated, info = self.env.step(action)
        return state, reward, terminated, truncated, info

    def reset(self):
        self.env.reset()
        self.l1_cache.clear()
        self.net_init(self.actor_net, self.value_net, self.actor_net_kwargs, self.value_net_kwargs)

    def episode_learn(self) -> Tuple[float, int]:
        episode_reward = 0
        state, info = self.env.reset()
        # frame_idx - counting steps
        self.frame_idx = -1
        done = False
        while not done:
            self.frame_idx += 1
            action, a_logprob = self.select_action(state, info, eps_threshold=None)

            # step
            next_state, reward, terminated, truncated, info = self.step(state, action)
            episode_reward += reward

            done = terminated or truncated

            # if terminated:
            #     next_state = None
            # else:
            #     next_state = state

            state = next_state
            # update the timesteps +1
            self.RLSYNC_obj.add_time_step()
            self.local_timestep += 1

            # training step
            self.train_step(state, action, a_logprob, reward, next_state, flush=done)

            if done:
                if truncated:
                    self.RLSYNC_obj.add_truncated()
                elif terminated:
                    self.RLSYNC_obj.add_terminated()
                break
        return episode_reward, self.frame_idx + 1

    def train_step(self, state, action, a_logprob, reward, next_state, flush=False):
        # Creating L1 cache to unsync agents with replay buffer
        self.l1_cache.append(PPOTransition(state, action, a_logprob, reward, next_state))
        if flush:
            self.local_episode += 1
            self.memory.extend_episode(self.l1_cache)
            if self.local_episode % self.ConfigAgent.SYNC == 0:
                if RLSYNC_obj.get_agents_running() > 1:
                    self.optimize_model()
                else:
                    self.optimize_model()
                    self.memory.clear()
            self.l1_cache.clear()
            self.soft_update()

    def optimize_model(self):
        """
        Args:
        """
        if self.ConfigAgent.BATCH_SIZE * self.RLSYNC_obj.get_agents_running() >= len(self.memory.episodes_indexes):
            s, a, r, s_, old_a_logprob = [], [], [], [], []
            for transition in self.memory.get_episodes_range(
                    self.ConfigAgent.BATCH_SIZE * self.RLSYNC_obj.get_agents_running()):
                s.append(transition.state)
                a.append(transition.action)
                r.append([transition.reward])
                s_.append(transition.next_state)
                old_a_logprob.append(transition.a_logprob)

            s = torch.from_numpy(np.array(s)).to(self.device)
            a = torch.from_numpy(np.array(a)).to(self.device)
            r = torch.from_numpy(np.array(r, dtype=np.float32)).to(self.device).view(-1, 1)
            s_ = torch.from_numpy(np.array(s_)).to(self.device)
            old_a_logprob = torch.from_numpy(np.array(old_a_logprob)).to(self.device).view(-1, 1)

            with torch.no_grad():
                target_v = r + self.ConfigAgent.GAMMA * self.actor_net(s_)[1]
                adv = target_v - self.actor_net(s)[1]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            for _ in range(self.n_epochs):
                for index in BatchSampler(SubsetRandomSampler(range(s.shape[0])), self.batch_size, False):
                    mu, sigma = self.actor_net(s[index])[0]
                    dist = torch.distributions.Normal(mu, sigma)
                    a_logprob = dist.log_prob(a[index]).sum(dim=1, keepdim=True)

                    ratio = torch.exp(a_logprob - old_a_logprob[index])

                    surrogate_loss1 = ratio * adv[index]
                    surrogate_loss2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                    policy_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()
                    value_loss = F.smooth_l1_loss(self.actor_net(s[index])[1], target_v[index])
                    # value_loss = F.mse_loss(r[index], values_pred)
                    entropy_loss = -torch.mean(-a_logprob)
                    total_loss = policy_loss + value_loss * self.vf_coef + entropy_loss * self.ent_coef

                    self.actor_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

    def soft_update(self):
        if self.RLSYNC_obj.get_agents_running() > 1:
            self.save_transfer_weights()
            if self.local_episode % self.ConfigAgent.AGENTS_SYNC == 0:
                # θ′ ← τ θ + (1 −τ )θ′
                self.agents_net_updates()

    def agents_net_updates(self):
        def state_dict_update(local_dict, other_dict):
            for key in local_dict:
                local_dict[key] = other_dict[key] * a_TAU + local_dict[key] * (1 - a_TAU)
            return local_dict

        a_TAU = self.ConfigAgent.TAU
        agents_lst = list(range(self.RLSYNC_obj.get_agents_running()))
        agents_lst.remove(self.id_num)

        local_actor_net_state_dict = copy.deepcopy(self.actor_net.state_dict())
        local_value_net_state_dict = {}

        if self.discrete:
            local_value_net_state_dict = copy.deepcopy(self.value_net.state_dict())

        for agent_id in agents_lst:
            self.load_transfer_weights(agent_id)
            other_actor_net_state_dict = self.actor_net.state_dict()
            self.actor_net.load_state_dict(state_dict_update(local_actor_net_state_dict, other_actor_net_state_dict))

            if self.discrete:
                other_value_net_state_dict = self.value_net.state_dict()
                self.value_net.load_state_dict(
                    state_dict_update(local_value_net_state_dict, other_value_net_state_dict))

    def transform_reward(self, state, observation, reward, terminated, truncated, info):
        pass

    def get_frames_sequence(self, render=True):
        if self.use_env_wrapper:
            return self._get_stacked_frames_sequence(render=True)
        else:
            super().get_frames_sequence(render)

    def _get_stacked_frames_sequence(self, render=True):
        frames = []
        # terminated = 0
        state, info = self.env.reset()
        episode_reward = 0.
        done = False
        while not done:
            action = self.get_action(state, info)
            next_state, reward, terminated, truncated, info = self.step(state, action)
            episode_reward += reward
            if render:
                frames.extend(self.env.render())
            else:
                frames.append(1)
            done = terminated or truncated
            state = next_state
        return frames, episode_reward, done

    def load(self, pathname: str, weights_only=False):
        checkpoint_num = pathname.split('-')[-1]
        path_filename_wo_ext = os.path.join(pathname, f'{self.agent_algo}-{checkpoint_num}')
        self.load_weights(path_filename_wo_ext)
        if not self.memory.ready:
            if not weights_only:
                self.memory.load(f'{path_filename_wo_ext}.pkl')
        self.save_transfer_weights()

    def save(self, pathname: str, checkpoint_num: int):
        path_filename_wo_ext = os.path.join(pathname, f'{self.agent_algo}-{checkpoint_num}')
        self.save_weights(f'{path_filename_wo_ext}')
        self.memory.save(f'{path_filename_wo_ext}.pkl')

    def save_weights(self, path_filename_wo_ext: str):
        torch.save(self.actor_net.state_dict(), f'{path_filename_wo_ext}_actor.pth')
        if self.discrete:
            torch.save(self.value_net.state_dict(), f'{path_filename_wo_ext}_value.pth')

    def load_weights(self, path_filename_wo_ext: str):
        if self.RLSYNC_obj is None:
            self.RLSYNC_obj = RLSYNC_obj

        def get_weights(path_filename):
            try:
                __weights = torch.load(path_filename, map_location=self.device)
            except:
                __weights = self.prepared_for_device(torch.load(path_filename))
            return __weights

        self.actor_net.load_state_dict(get_weights(f'{path_filename_wo_ext}_actor.pth'))
        if self.discrete:
            self.value_net.load_state_dict(get_weights(f'{path_filename_wo_ext}_value.pth'))

    def save_transfer_weights(self):
        if self.discrete:
            RLSYNC_obj.save_weights(self.id_num, [copy.deepcopy(self.actor_net.state_dict()),
                                                  copy.deepcopy(self.value_net.state_dict())])
        else:
            RLSYNC_obj.save_weights(self.id_num, [copy.deepcopy(self.actor_net.state_dict())])

    def load_transfer_weights(self, agent_id):
        __weights = self.RLSYNC_obj.get_weights(agent_id)
        with self.RLSYNC_obj.lock:
            try:
                self.actor_net.load_state_dict(__weights[0])
            except:
                self.actor_net.load_state_dict(self.prepared_for_device(__weights[0]))
            if self.discrete:
                try:
                    self.value_net.load_state_dict(__weights[1])
                except:
                    self.value_net.load_state_dict(self.prepared_for_device(__weights[1]))
