import io
import copy
import math
import random
import imageio
import numpy as np
from itertools import count

import torch
import torch.optim as optim

from rlcollection.rlnetworks import FCnet
from rlcollection.replaybuffer import Transition
from rlcollection.configdqn import ConfiqDQN
from rlcollection.rlsync import RLSYNC_obj

__version__ = 0.013


class DQNAgent:
    """Interacts with and learns form environment."""
    id_count: int = 0

    def __init__(self, env, seed, device='cpu', net_model=FCnet):
        """
        Args:
            env:            initialized environment class
            seed (int):     random seed
        """
        self.id_num = int(DQNAgent.id_count)
        DQNAgent.id_count += 1

        self.device = device
        self.env = copy.deepcopy(env)
        self.seed: int = seed
        self.net_model = net_model
        random.seed(self.seed)

        self.__RLSYNC_obj = None
        self.memory = None

        self.state_size = len(self.env.observation_space.high)
        self.n_actions = self.env.action_space.n
        self.policy_net = None
        self.target_net = None
        self.proxy_net = None
        self.optimizer = None
        self.l1_cache_size = ConfiqDQN.BATCH_SIZE * 3
        self.l1_cache: list = []


    @property
    def RLSYNC_obj(self):
        return self.__RLSYNC_obj

    @RLSYNC_obj.setter
    def RLSYNC_obj(self, obj):
        self.__RLSYNC_obj = obj
        #   setting shared mp or multithreaded variables through RLSYNC_obj
        self.memory = self.__RLSYNC_obj.memory
        self.memory.buffer_resize(ConfiqDQN.BUFFER_SIZE)
        self.net_init(self.net_model, self.net_model)

    def net_init(self, policy_net, target_net):
        with self.__RLSYNC_obj.lock:
            self.policy_net = policy_net(self.state_size, self.n_actions, self.seed).to(self.device)
            self.target_net = target_net(self.state_size, self.n_actions, self.seed).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=ConfiqDQN.LR, amsgrad=True)
            self.save_transfer_weights()

    def select_action(self, state):
        """
        Returns action for discrete action space with epsilon greedy action selection to explore space

        Args:
            state (np.array):   state (observations) as np.array
        Returns:
            action:     action
        """
        # Epsilon greedy action selection
        eps_threshold = ConfiqDQN.EPS_END + (ConfiqDQN.EPS_START - ConfiqDQN.EPS_END) * math.exp(
            -1. * self.RLSYNC_obj.sync_time_step / ConfiqDQN.EPS_DECAY)

        # update the timesteps
        with self.RLSYNC_obj.lock:
            self.RLSYNC_obj.sync_time_step += 1

        if random.random() > eps_threshold:
            return self.get_action(state)
        else:
            return self.env.action_space.sample()

    def get_action(self, state):
        """
        Returns action for discrete action space from learned network

        Args:
            state (np.array):   state (observations) as np.array
        Returns:
            action:     action
        """
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # t.max(1) вернет наибольшее значение столбца в каждой строке.
            # Второй столбец в результате max - это индекс того места,
            # где был найден максимальный элемент,
            # поэтому мы выбираем действие с наибольшим ожидаемым вознаграждением.
            return self.policy_net(state).max(1)[1].view(1, 1).item()

    def optimize_model(self, batch_size):
        """

        Args:
            batch_size(int):    batch_size
        """
        experience = self.memory.sample(batch_size)
        batch = Transition(*zip(*experience))

        _non_final_mask = np.array(tuple(map(lambda s: s is not None, batch.next_state)))
        _non_final_next_states = np.array([s for s in batch.next_state if s is not None])
        _s_batch = np.array(batch.state)
        _a_batch = np.array(batch.action)
        _r_batch = np.array(batch.reward)

        def calc_gradient_algo(state_batch, action_batch, reward_batch):
            # Вычислить маску нефинальных состояний и соединить элементы батча
            # (финальным состоянием должно быть то, после которого моделирование закончилось)
            t_non_final_mask = torch.tensor(_non_final_mask, device=self.device,
                                            dtype=torch.bool)

            t_non_final_next_states = torch.tensor(_non_final_next_states, device=self.device)

            # Собираем батчи для состояний, действий и наград
            state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)

            # Вычислить Q(s_t, a) - модель вычисляет Q(s_t),
            # затем мы выбираем столбцы предпринятых действий.
            # Это те действия, которые были бы предприняты для каждого состояния партии в соответствии с policy_net
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Вычислить V(s_{t+1}) для всех следующих состояний.
            # Ожидаемые значения действий для не_финальных_следующих_состояний вычисляются
            # на основе "старшей" целевой_сети; выбирается их наилучшее вознаграждение с помощью max(1)[0].
            # Это объединяется по маске, так что мы будем иметь либо ожидаемое значение состояния,
            # либо 0, если состояние было финальным.
            next_state_values = torch.zeros(batch_size, device=self.device)
            with torch.no_grad():
                next_state_values[t_non_final_mask] = self.target_net(t_non_final_next_states).max(1)[0]
            # Вычисляем ожидаемые Q значения
            expected_state_action_values = (next_state_values * ConfiqDQN.GAMMA) + reward_batch

            # Объединяем все в общий лосс
            criterion = torch.nn.HuberLoss()
            # criterion = torch.nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Готовим градиент
            self.optimizer.zero_grad()
            loss.backward()
            # Обрезаем значения градиента - проблемма исчезающего/взрывающего градиента
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

            self.optimizer.step()

        calc_gradient_algo(_s_batch, _a_batch, _r_batch)
        # update target net if necessary
        self.soft_update()

    def target_net_update(self):
        policy_net_state_dict = self.policy_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * ConfiqDQN.TAU + target_net_state_dict[
                key] * (1 - ConfiqDQN.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def agents_net_updates(self):
        agents_lst = list(range(self.RLSYNC_obj.agents_running))
        agents_lst.remove(self.id_num)

        local_policy_net_state_dict = self.policy_net.state_dict()
        local_target_net_state_dict = self.target_net.state_dict()

        for agent_id in agents_lst:
            self.load_transfer_weights(agent_id)
            other_policy_net_state_dict = self.policy_net.state_dict()
            other_target_net_state_dict = self.target_net.state_dict()
            for key in other_policy_net_state_dict:
                local_policy_net_state_dict[key] = other_policy_net_state_dict[key] * ConfiqDQN.TAU + \
                                                   local_policy_net_state_dict[key] * (1 - ConfiqDQN.TAU)
            self.policy_net.load_state_dict(local_policy_net_state_dict)

            for key in other_target_net_state_dict:
                local_target_net_state_dict[key] = other_target_net_state_dict[key] * ConfiqDQN.TAU + \
                                                   local_target_net_state_dict[key] * (1 - ConfiqDQN.TAU)
            self.target_net.load_state_dict(local_target_net_state_dict)

    def soft_update(self):
        """  Soft update model parameters """
        if self.RLSYNC_obj.sync_time_step % ConfiqDQN.SYNC_FRAME == 0:
            # θ′ ← τ θ + (1 −τ )θ′
            self.target_net_update()
        if (self.RLSYNC_obj.agents_running > 1) and (self.RLSYNC_obj.sync_time_step >= ConfiqDQN.AGENTS_SYNC_FRAME):
            self.save_transfer_weights()
            if self.RLSYNC_obj.sync_time_step % ConfiqDQN.AGENTS_SYNC_FRAME == 0:
                # θ′ ← τ θ + (1 −τ )θ′
                self.agents_net_updates()

    def agent_learn(self, rlsync_obj=None):
        if rlsync_obj is None:
            self.RLSYNC_obj = RLSYNC_obj
        else:
            self.RLSYNC_obj = rlsync_obj

        while self.RLSYNC_obj.sync_time_step <= self.RLSYNC_obj.sync_total_time_steps:
            episode_reward, episode_length = self.episode_learn()
            with self.RLSYNC_obj.lock:
                self.RLSYNC_obj.episodes_rewards.append(episode_reward)
                self.RLSYNC_obj.episodes_length.append(episode_length)

    def episode_learn(self) -> tuple:
        episode_reward = 0

        # Для каждого эпизода инициализируем начальное состояние
        state, info = self.env.reset()

        # frame_idx - считает сколько шагов успели сделать пока не приземлились
        for frame_idx in count():
            # выбираем действие [0, 1]
            action = self.select_action(state)
            # Делаем шаг
            observation, reward, terminated, truncated, _ = self.env.step(action)
            episode_reward += reward

            # Объединяем done по двум конечным состояниям
            done = terminated or truncated
            # присваиваем следующее состояние
            if terminated:
                next_state = None
            else:
                next_state = observation
            # making a step in learning
            self.step(state, [action], next_state, reward, flush=done)
            # переходим на следующее состояние
            state = next_state
            if done:
                break

        return episode_reward, frame_idx + 1

    def step(self, state, action, reward, next_state, flush=False):
        # Creating L1 cache to unsync agents with replay buffer
        self.l1_cache.append([state, action, reward, next_state])
        if flush or (len(self.l1_cache) >= random.randint(ConfiqDQN.BATCH_SIZE, self.l1_cache_size)):
            self.memory.extend([Transition(*element) for element in self.l1_cache])
            self.l1_cache.clear()

        if len(self.memory) > ConfiqDQN.BATCH_SIZE:
            self.optimize_model(ConfiqDQN.BATCH_SIZE)

    def save(self, path_filename: str):
        torch.save(self.policy_net.state_dict(), path_filename)

    def load(self, path_filename: str):
        self.policy_net.load_state_dict(torch.load(path_filename, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_transfer_weights(self):
        RLSYNC_obj.save_weights(self.id_num, [self.policy_net.state_dict(), self.target_net.state_dict()])

    def load_transfer_weights(self, agent_id):
        def prepared_for_device(state_dict):
            temp_buffer = io.BytesIO()
            torch.save(state_dict, temp_buffer)
            temp_buffer.seek(0)
            return torch.load(temp_buffer, map_location=self.device)

        __weights = self.RLSYNC_obj.get_weights(agent_id)
        self.policy_net.load_state_dict(prepared_for_device(__weights[0]))
        self.target_net.load_state_dict(prepared_for_device(__weights[1]))

    def save_movie_gif(self, gif_path_filename: str):
        frames = []
        state, info = self.env.reset()
        episode_reward = 0.
        done = False
        while not done:
            action = self.get_action(state)
            state, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            if any((terminated, truncated)):
                frames = self.env.render()
                done = True
        saveanimation(frames, gif_path_filename)
        return episode_reward, len(frames)

    def reset(self):
        self.l1_cache.clear()
        self.net_init(self.net_model, self.net_model)


def saveanimation(frames, _path_filename="./movie.gif"):
    """
    This method ,given the frames of images make the gif and save it in the folder

    params:
        frames:method takes in the array or np.array of images
        address:(optional)given the address/location saves the gif on that location
                otherwise save it to default address './movie.gif'

    return :
        none
    """
    imageio.mimsave(_path_filename, frames)
