import gymnasium as gym
import numpy as np
import random
import math


class RobustBayesianTabularDQNAgent:
    def __init__(self, env, learning_rate=0.1, gamma=0.9, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.95):
        self.env = env
        self.learning_rate = learning_rate  # Скорость обучения
        self.gamma = gamma  # Коэффициент дисконтирования
        self.epsilon = epsilon_start  # Начальное значение эпсилона
        self.epsilon_min = epsilon_min  # Минимальное значение эпсилона
        self.epsilon_decay = epsilon_decay  # Темп уменьшения эпсилона

        # Размеры для таблиц Q и N
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Инициализируем таблицы для Q-значений и количества посещений состояний
        self.Q_table = np.zeros((num_states, num_actions))  # Q(s, a)
        self.N_table = np.zeros((num_states, num_actions), dtype=np.int32)  # N(s, a)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])  # Выбираем действие с максимальным Q-значением

    def get_action(self, state):
        return np.argmax(self.Q_table[state])

    def update_Q_value(self, state, action, reward, next_state, done):
        # Обновление Q-значения
        current_q = self.Q_table[state, action]
        next_max_q = np.max(self.Q_table[next_state]) if not done else 0
        target = reward + self.gamma * next_max_q
        self.Q_table[state, action] += self.learning_rate * (target - current_q)

        # Увеличение счетчика посещений состояния-действия
        self.N_table[state, action] += 1

        # Байесовская корректировка уверенности
        n = self.N_table[state, action]
        variance = 1 / (1 + n)  # Упрощённая оценка дисперсии
        confidence = math.sqrt(variance)  # Оценка стандартной ошибки
        adjusted_q = self.Q_table[state, action] + confidence  # Учёт неопределённости
        self.Q_table[state, action] = adjusted_q

    def train(self, episodes=10000, retrain_every=1000):
        rewards_history = []
        max_reward = -np.inf
        for episode in range(episodes):
            state, info = self.env.reset()
            total_reward = 0
            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.update_Q_value(state, action, reward, next_state, terminated or truncated)
                state = next_state
                total_reward += reward
                if terminated or truncated:
                    break
            rewards_history.append(total_reward)
            # Уменьшение значения эпсилона
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Периодическое переподучение
            # if (episode + 1) % retrain_every == 0:
            #     print(f"Episode: {episode}, Retraining...")
            #     self.retrain()
            print(f"\rEpisode: {episode}, Total Reward: {total_reward}, Max Reward: {max_reward}", end='')
            if total_reward > max_reward:
                max_reward = total_reward

        return rewards_history

    def set_env(self, _env):
        self.env = _env

    def evaluate(self):
        state, info = self.env.reset()
        total_reward = 0
        while True:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        print(f"\nEvaluation Reward: {total_reward}")


if __name__ == "__main__":
    from gymnasium.utils.save_video import save_video

    # env_id = "Taxi-v3"
    env_id = "FrozenLake-v1"

    env = gym.make(env_id)
    agent = RobustBayesianTabularDQNAgent(env)
    rewards_history = agent.train(episodes=30000)
    agent.evaluate()
    env.close()

    env = gym.make(env_id, render_mode="rgb_array_list")
    state, _ = env.reset()
    step_starting_index = 0
    episode_index = 0
    for step_index in range(199):
        action = agent.get_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state
        if terminated or truncated:
            save_video(
                frames=env.render(),
                video_folder="videos",
                fps=env.metadata["render_fps"],
                step_starting_index=step_starting_index,
                episode_index=episode_index
            )
            step_starting_index = step_index + 1
            episode_index += 1
            env.reset()
    env.close()
