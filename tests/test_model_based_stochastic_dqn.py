import datetime
import numpy as np
import pickle
import random
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from pytz import timezone

TZ = timezone('Europe/Moscow')

__version__ = 0.010


def get_exp_id() -> str:
    return f'exp-{datetime.datetime.now(TZ).strftime("%y%m%d-%H%M%S")}'


class StochasticModel:
    def __init__(self, env):
        self.real_env = env
        self.num_states = self.real_env.observation_space.n
        self.num_actions = self.real_env.action_space.n
        self.transition_counts = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.int32)

    def step(self, state, action):
        next_state, reward, terminated, truncated, info = self.real_env.step(action)
        self.update_transition_model(state, action, next_state)
        return next_state, reward, terminated, truncated, info

    def reset(self):
        return self.real_env.reset()

    def update_transition_model(self, state, action, next_state):
        self.transition_counts[state, action, next_state] += 1

    def get_real_env(self):
        return self.real_env

    def __get_state__(self):
        """Возвращает состояние объекта."""
        return {
            'transition_counts': self.transition_counts,
            'num_states': self.num_states,
            'num_actions': self.num_actions
        }

    def __set_state__(self, state_dict):
        """Устанавливает состояние объекта."""
        self.transition_counts = state_dict['transition_counts']
        self.num_states = state_dict['num_states']
        self.num_actions = state_dict['num_actions']

    def save_virtual_environment(self, filename):
        """Сохраняем виртуальную модель среды."""
        with open(filename, 'wb') as file:
            pickle.dump(self.__get_state__(), file)

    @classmethod
    def load_virtual_environment(cls, filename, real_env):
        """Загружаем сохраненную виртуальную модель среды."""
        with open(filename, 'rb') as file:
            state_dict = pickle.load(file)
        ve = cls(real_env)
        ve.__set_state__(state_dict)
        return ve


class TabularStochasticQAgent:
    def __init__(self, virtual_env, learning_rate=0.01, gamma=0.9, epsilon_start=1.0, epsilon_min=0.05):
        self.virtual_env = virtual_env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        # Размеры для таблиц Q и N
        self.num_states = virtual_env.num_states
        self.num_actions = virtual_env.num_actions

        # Инициализируем трехмерные таблицы для Q-значений и количества посещений состояний
        self.Q_table = np.zeros((self.num_states, self.num_actions, self.num_states))  # Q(s, a, s')
        self.N_table = self.virtual_env.transition_counts

        # # Инициализируем веса действий
        # self.w_table = np.zeros((num_states, num_actions))  # w(s, a)

    def _calculate_weighted_q_value(self, q_matrix, transition_matrix):
        # Суммы переходов для каждого действия
        total_transitions = np.sum(transition_matrix, axis=1)

        # Обработка нулевых значений в знаменателе
        # Если сумма переходов равна нулю, заменяем её на минимальное положительное число
        safe_total_transitions = np.where(total_transitions == 0, 1e-8, total_transitions)

        # Вероятности переходов для каждого состояния
        transition_probs = np.divide(transition_matrix, safe_total_transitions.reshape(-1, 1),
                                     where=(safe_total_transitions.reshape(-1, 1) != 0))

        # Маска для нулевых значений
        masked_q_matrix = np.ma.masked_where(transition_probs == 0, q_matrix)

        # Максимум по каждому столбцу, исключая нули
        weighted_q_values = np.ma.max(masked_q_matrix, axis=1)

        return weighted_q_values

    def select_action(self, state, info):
        if random.random() < self.epsilon:
            return self.virtual_env.real_env.action_space.sample()
            # return self.virtual_env.real_env.action_space.sample(info["action_mask"])
        else:
            return self.get_action(state)

    def get_action(self, state):
        # Рассчитаем взвешенные Q-значения для каждого действия
        weighted_q_values = self._calculate_weighted_q_value(self.Q_table[state], self.N_table[state])

        # Суммируем количество переходов по всем действиям
        total_transitions_sum = np.sum(self.N_table[state])

        # Рассчитаем веса действий с безопасным делением
        action_weights = np.divide(np.sum(self.N_table[state], axis=1),
                                   total_transitions_sum,
                                   where=(total_transitions_sum != 0))

        # Умножим взвешенные Q-значения на веса действий
        weighted_q_values_with_action_weights = weighted_q_values * action_weights

        # Найдем максимальное взвешенное Q-значение и соответствующее действие
        # если все Q_value не равны нулю
        if not np.any(weighted_q_values_with_action_weights):
            return self.virtual_env.real_env.action_space.sample()
        else:
            return np.argmax(weighted_q_values_with_action_weights)

    def update_Q_value(self, state, action, reward, next_state, done):
        # Рассчитываем целевое значение Q-функции
        if not done:
            future_q = np.max(self.Q_table[next_state])
        else:
            future_q = 0.0

        target = reward + self.gamma * future_q

        # Обновляем Q-значение
        current_q = self.Q_table[state, action, next_state]
        update_q = (self.learning_rate * (target - current_q))
        # backpropagation 1 step but for all possible next_states
        # we filter them with probability at select_action time
        self.Q_table[state, action] += update_q

    def train(self, episodes=10000, truncate_steps=200, stats=100, tb_log='TB'):
        writer = SummaryWriter(
            log_dir=f'{tb_log}/{self.virtual_env.real_env.spec.id}{get_exp_id()}')  # Создаем объект SummaryWriter
        rewards = deque(maxlen=stats)
        max_reward = -np.inf
        for episode in range(episodes):
            state, info = self.virtual_env.reset()
            total_reward = 0
            step_ix = 0
            while True:
                action = self.select_action(state, info)
                next_state, reward, terminated, truncated, info = self.virtual_env.step(state, action)
                truncated = (step_ix == truncate_steps)
                done = terminated or truncated
                self.update_Q_value(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
                step_ix += 1

            # Обновляем значение эпсилона в зависимости от текущего эпизода
            self.epsilon = max(self.epsilon_min,
                               self.epsilon_start - (self.epsilon_start - self.epsilon_min) * (
                                       (episode + 1) / episodes))
            rewards.append(total_reward)
            # save best weights
            if total_reward >= max_reward:
                max_reward = total_reward
                self.save_agent(f'{env_id}_agent.pkl')
                self.virtual_env.save_virtual_environment(f'{env_id}_ve.pkl')

            writer.add_scalar("Reward/Total", total_reward, episode)
            writer.add_scalar("Learning rate", self.learning_rate, episode)
            writer.add_scalar("Max Reward", max_reward, episode)
            writer.add_scalar(f"Mean Rolling Reward {stats}", np.mean(rewards), episode)
            writer.add_scalar("Epsilon", self.epsilon, episode)

            print(
                f"\rEpisode: {episode}, Total Reward: {total_reward}, Max episode reward: {max_reward}, Epsilon: {self.epsilon:.5f}",
                end='')

        writer.close()

    def __getstate__(self):
        """Возвращает состояние объекта."""
        return self.__dict__

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)

    def save_agent(self, filename):
        """Сохраняем агента."""
        with open(filename, 'wb') as file:
            pickle.dump(self.__getstate__(), file)

    @classmethod
    def load_agent(cls, filename):
        """Загружаем сохраненного агента."""
        with open(filename, 'rb') as file:
            state_dict = pickle.load(file)
        agent = cls(state_dict['virtual_env'])
        agent.__setstate__(state_dict)
        return agent


if __name__ == "__main__":
    from gymnasium.utils.save_video import save_video

    # env_id = "Taxi-v3"
    # truncate_steps = 200
    # env_kwargs = dict(id=env_id)
    # agent_kwargs = dict(learning_rate=0.05, gamma=0.99)
    # train_kwargs = dict(episodes=10000, truncate_steps=truncate_steps, tb_log='TB')

    # from gymnasium.envs.toy_text.frozen_lake import generate_random_map

    env_id = "FrozenLake-v1"
    truncate_steps = 100
    # desc = generate_random_map(size=6)
    # print(desc)
    # desc = ['SFFFHF', 'FFFFFF', 'FFHHHF', 'HFFFFH', 'FFFFHF', 'FFFFFG']
    desc = ['SFFHF', 'FFFFF', 'FHFHF', 'FFFHF', 'FFFFG']
    map_name = f'{len(desc)}x{len(desc)}'
    env_kwargs = dict(id=env_id,
                      desc=desc,
                      is_slippery=True,
                      map_name=map_name
                      )
    agent_kwargs = dict(learning_rate=0.1, gamma=0.99)
    train_kwargs = dict(episodes=15000, truncate_steps=truncate_steps, tb_log='TB')

    real_env = gym.make(**env_kwargs)
    virtual_env = StochasticModel(real_env)
    agent = TabularStochasticQAgent(virtual_env, **agent_kwargs)
    agent.train(**train_kwargs)

    # Загрузка агента и виртуальной среды
    loaded_agent = TabularStochasticQAgent.load_agent(f'{env_id}_agent.pkl')

    real_env.close()

    env_kwargs.update({'render_mode': 'rgb_array_list'})
    env = gym.make(**env_kwargs)
    loaded_agent.real_env = env
    for episode_index in range(5):
        # noinspection PyRedeclaration
        state, _ = env.reset()
        for step_index in range(truncate_steps):
            action = loaded_agent.get_action(state)
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            if terminated or truncated:
                save_video(
                    frames=env.render(),
                    video_folder="videos",
                    fps=env.metadata["render_fps"],
                    episode_trigger=lambda x: True,
                    episode_index=episode_index,
                    name_prefix=f'{env_id}_rl_video'
                )
                break
    env.close()
