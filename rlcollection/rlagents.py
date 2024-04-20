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
from rlcollection.rlsync import SYNC_obj

__version__ = 0.010


class DQNAgent:
    """Interacts with and learns form environment."""
    id_count: int = 0

    def __init__(self, env, seed, device='cpu', net_model=FCnet):
        """
        Args:
            env:            initialized environment class
            seed (int):     random seed
        """
        DQNAgent.id_count += 1
        self.id_num = int(DQNAgent.id_count)

        self.device = device
        self.env = copy.deepcopy(env)
        self.seed = seed
        self.net_model = net_model
        random.seed(self.seed)

        #   setting shared mp or multithreaded variables through SYNC_obj
        self.memory = SYNC_obj.memory
        self.memory.buffer_resize(ConfiqDQN.BUFFER_SIZE)

        self.state_size = len(self.env.observation_space.high)
        self.n_actions = self.env.action_space.n
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        if device == 'cuda':
            with SYNC_obj.lock:
                self.net_init(self.net_model, self.net_model)
        else:
            self.net_init(self.net_model, self.net_model)

    def net_init(self, policy_net, target_net):
        self.policy_net = policy_net(self.state_size, self.n_actions, self.seed).to(self.device)
        self.target_net = target_net(self.state_size, self.n_actions, self.seed).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=ConfiqDQN.LR, amsgrad=True)

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
            -1. * SYNC_obj.sync_time_step / ConfiqDQN.EPS_DECAY)

        # update the timesteps
        with SYNC_obj.lock:
            SYNC_obj.sync_time_step += 1

        if random.random() > eps_threshold:
            if self.device == 'cuda':
                with SYNC_obj.lock:
                    return self.get_action(state)
            else:
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
        s_batch = np.array(batch.state)
        a_batch = np.array(batch.action)
        r_batch = np.array(batch.reward)

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

        if self.device == 'cuda':
            with SYNC_obj.lock:
                calc_gradient_algo(s_batch, a_batch, r_batch)
        else:
            calc_gradient_algo(s_batch, a_batch, r_batch)
        # update target net if necessary
        self.soft_update()

    def target_net_update(self):
        policy_net_state_dict = self.policy_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * ConfiqDQN.TAU + target_net_state_dict[
                key] * (1 - ConfiqDQN.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def soft_update(self):
        """  Soft update model parameters """
        if SYNC_obj.sync_time_step % ConfiqDQN.SYNC_FRAME == 0:
            # θ′ ← τ θ + (1 −τ )θ′
            if self.device == 'cuda':
                with SYNC_obj.lock:
                    self.target_net_update()
            else:
                self.target_net_update()

    def agent_learn(self) -> tuple:
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
            self.step(state, [action], next_state, reward)
            # переходим на следующее состояние
            state = next_state
            if done:
                break
        return episode_reward, frame_idx + 1

    def step(self, state, action, reward, next_state, ):
        # Save experience in replay buffer
        self.memory.add(state, action, reward, next_state)

        if len(self.memory) > ConfiqDQN.BATCH_SIZE:
            self.optimize_model(ConfiqDQN.BATCH_SIZE)

    def save(self, path_filename: str):
        torch.save(self.policy_net.state_dict(), path_filename)

    def load(self, path_filename: str):
        self.policy_net.load_state_dict(torch.load(path_filename, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

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
