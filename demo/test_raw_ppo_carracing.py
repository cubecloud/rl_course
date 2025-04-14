import sys
import os

import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
import torch.nn.functional as F
# import cv2
from collections import deque
import matplotlib.pyplot as plt


class FrameStack:
    def __init__(self, num_frames: int = 4, frame_shape=(84, 84), device='cuda'):
        self.frame_shape = frame_shape
        self.frames = deque(maxlen=num_frames)
        self.device = device

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    def _preprocess_frame(self, frame) -> torch.Tensor:
        # converting from rgb to grayscale & expanding dims for interpolate
        frame = self.rgb2gray(frame).astype(np.float32)
        frame = torch.from_numpy(frame).unsqueeze(0)
        # resize to 84 x 84
        # plt.imshow(frame.squeeze(0).squeeze(0), cmap=plt.get_cmap('summer'), vmin=0, vmax=1)
        # frame = F.interpolate(torch.from_numpy(frame), size=self.frame_shape, mode='nearest').squeeze(0)
        # img = mpimg.imread('image.png')
        # plt.imshow(frame.squeeze(0).squeeze(0), cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        # plt.show()
        # sys.exit()
        return frame

    def __call__(self, frame) -> torch.Tensor:
        self.add_frame(frame)
        while len(self.frames) < self.frames.maxlen:
            self.frames.append(self.frames[0])
        return self.get_stacked_frames()

    def add_frame(self, frame):
        self.frames.appendleft(self._preprocess_frame(frame))

    def get_stacked_frames(self) -> torch.Tensor:
        return torch.cat(list(self.frames)).unsqueeze(0).to(self.device)

    def reset(self):
        self.frames.clear()


# 1. Определение модели политики (нейронная сеть)
class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, input_size, action_dim):
        super(ContinuousPolicyNetwork, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(4, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)

        # Calculate combined feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 4, input_size, input_size)
            d = self.cnn_base(dummy_input)
            d = d.view(d.size(0), -1)
            conv2d_features = d.shape[-1]
            print(f'Shape of feature extractor: {conv2d_features}')
        self.fc = nn.Sequential(nn.Linear(conv2d_features, 100), nn.ReLU())

        with torch.no_grad():
            d = self.fc(d)
            self._features_dim = d.shape[-1]
            print(f'features_dim: {d.shape[-1]}')

        # Head для политики (распределение действий)
        # self.mu_layer = nn.Linear(self._features_dim, action_dim)
        # self.sigma_layer = nn.Linear(self._features_dim, action_dim)
        self.mu_layer = nn.Sequential(nn.Linear(self._features_dim, action_dim), nn.Tanh())
        self.sigma_layer = nn.Sequential(nn.Linear(self._features_dim, action_dim), nn.Softplus())
        self.vf_layer = nn.Sequential(nn.Linear(self._features_dim, self._features_dim),
                                      nn.ReLU(),
                                      nn.Linear(self._features_dim, 1))
        self.apply(self._weights_init)


    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mu = self.mu_layer(x)  # Ограничиваем выход му для диапазона [-1, 1]
        sigma = self.sigma_layer(x) + 1e-5  # Гарантируем положительность сигмы
        v = self.vf_layer(x)
        return (mu, sigma), v


# 2. Функция выбора действий на основе текущей политики
def select_continuous_action(state, policy_network):
    # state = torch.from_numpy(state).float().unsqueeze(0)
    with torch.no_grad():
        (mu, sigma), value = policy_network(state)
    normal_distribution = torch.distributions.Normal(mu, sigma)
    action = normal_distribution.sample()
    log_prob = normal_distribution.log_prob(action)
    return action.squeeze(0).cpu().numpy(), log_prob.sum(dim=1).item(), value.detach().cpu().numpy()


# 3. Класс буферизации переходов среды (опыт)
class Buffer:
    def __init__(self, gamma=0.99, lamda=0.95, episodes_capacity: int = 6):
        self.gamma = gamma
        self.lamda = lamda
        self.states, self.actions, self.rewards, self.mus, self.sigmas, self.logprobs, self.values, self.masks = [], [], [], [], [], [], [], []
        self.returns, self.advantages = [], []
        self.episodes_capacity = episodes_capacity
        self.episodes_num: int = -1
        self.episodes_length: list = []
        self.last_episode_length: int = 0

    def remove_oldest_episode(self):
        old_ep_len = self.episodes_length.pop(0)
        self.states = self.states[old_ep_len:]
        self.actions = self.actions[old_ep_len:]
        self.rewards = self.rewards[old_ep_len:]
        # self.mus = self.mus[old_ep_len:]
        # self.sigmas = self.sigmas[old_ep_len:]
        self.logprobs = self.logprobs[old_ep_len:]
        self.values = self.values[old_ep_len:]
        self.masks = self.masks[old_ep_len:]
        self.returns = self.returns[old_ep_len:]
        self.advantages = self.advantages[old_ep_len:]

        self.episodes_num -= 1

    def store(self, state, action, reward, mu, sigma, logprob, value, mask):
        self.last_episode_length += 1

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        # self.mus.append(mu)
        # self.sigmas.append(sigma)
        self.values.append(value)
        self.masks.append(mask)
        self.logprobs.append(logprob)

        # episode completed
        if not mask:
            self.episodes_num += 1
            self.episodes_length.append(self.last_episode_length)
            self.last_episode_length = 0
            if self.episodes_num > self.episodes_capacity:
                self.remove_oldest_episode()

    def compute_last_episode(self):
        ep_len = self.episodes_length[-1]
        # Преобразование списков в тензоры
        # states = torch.tensor(np.array(self.states[-ep_len:]), dtype=torch.float)
        # actions = torch.tensor(np.array(self.actions[-ep_len:]), dtype=torch.float)
        rewards = torch.tensor(np.array(self.rewards[-ep_len:]), dtype=torch.float)
        # mus = torch.tensor(np.array(self.mus), dtype=torch.float)
        # sigmas = torch.tensor(np.array(self.sigmas), dtype=torch.float)
        values = torch.tensor(np.array(self.values[-ep_len:]), dtype=torch.float)
        masks = torch.tensor(np.array(self.masks[-ep_len:]), dtype=torch.bool)
        # logprobs = torch.stack(self.logprobs[-ep_len:])

        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        # Вычисление advantage и return
        running_return = 0
        previous_value = 0
        previous_advantage = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * masks[t]
            delta = rewards[t] + self.gamma * previous_value * masks[t] - values[t]
            advantage = delta + self.gamma * self.lamda * previous_advantage * masks[t]

            returns[t] = running_return
            previous_value = values[t]
            previous_advantage = advantage
            advantages[t] = advantage

        # Нормализация преимуществ
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # self.states[-ep_len:] = states
        # self.actions[-ep_len:] = actions
        # self.logprobs[-ep_len:] = logprobs
        self.returns.extend(returns)
        self.advantages.extend(advantages)

    def compute_returns_and_advantages(self):
        self.compute_last_episode()

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        # del self.mus[:]
        # del self.sigmas[:]
        del self.logprobs[:]
        del self.values[:]
        del self.masks[:]
        del self.returns[:]
        del self.advantages[:]
        self.episodes_num: int = 0
        self.episodes_length: list = []
        self.last_episode_length: int = 0

# def calculate_value_loss(values, target_values):
#     value_loss = F.mse_loss(values, target_values)
#     return value_loss
#
#
def calculate_entropy_bonus(entropies):
    entropy_bonus = entropies.mean()
    return entropy_bonus


def calculate_total_loss(policy_loss, value_loss, entropy_bonus):
    total_loss = policy_loss + value_loss - entropy_bonus
    return total_loss


def calculate_policy_loss(new_logprobs, old_logprobs, advantages, epsilon):
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surrogate_1 = ratio * advantages
    surrogate_2 = clipped_ratio * advantages
    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
    return policy_loss


# 5. Алгоритм PPO
def ppo_update(policy_network, optimizer, buffer, K_epochs, eps_clip, device):
    for _ in range(K_epochs):

        start_idx = 0
        end_idx = 0

        for ep_len in buffer.episodes_length:
            end_idx += (ep_len + 1)
            # Выбираем данные из буфера
            states = torch.cat(buffer.states[start_idx: end_idx]).to(device)
            actions = torch.tensor(np.array(buffer.actions[start_idx: end_idx])).to(device)
            old_logprobs = torch.tensor(buffer.logprobs[start_idx: end_idx]).detach().to(device)
            returns = torch.tensor(buffer.returns[start_idx: end_idx]).detach().to(device)
            advantages = torch.tensor(buffer.advantages[start_idx: end_idx]).detach().to(device)

            mus, sigmas, values = policy_network(states)

            normal_distribution = Normal(mus, sigmas)
            new_logprobs = normal_distribution.log_prob(actions).sum(dim=-1)

            ratios = torch.exp(new_logprobs - old_logprobs)
            # clipped surrogate loss
            policy_loss_1 = ratios * advantages
            policy_loss_2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), returns)
            entropy_bonus = calculate_entropy_bonus(normal_distribution.entropy())
            total_loss = calculate_total_loss(policy_loss, value_loss, entropy_bonus)

            # Обновление весов сети
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            start_idx = end_idx


fms = FrameStack(num_frames=4)


# 6. Обучение агента
def train(env_name, max_episodes, max_timesteps, update_timestep, batch_size, K_epochs, gamma, lamda, eps_clip,
          policy_network, optimizer, device='cuda'):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    buffer = Buffer(gamma, lamda)
    episode_rewards = []
    average_rewards = []
    ppo_updates_num = 0

    for ep in range(max_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        timestep = 0
        while not done:
            # Генерируем действие и выполняем его в среде
            state = fms(state)
            action, logprob, value = select_continuous_action(state, policy_network)
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            # if total_reward < -50.0:
            #     truncated = True

            done = terminated or truncated

            # store step data to buffer
            buffer.store(state, action, reward, None, None, logprob, value, 1 - done)

            state = next_state

            timestep += 1

            # Проверяем, достигли ли мы порога обновления
            if done:
                buffer.compute_returns_and_advantages()

            if buffer.episodes_num >= buffer.episodes_capacity:
                ppo_update(policy_network, optimizer, buffer, K_epochs, eps_clip, device)
                buffer.clear()
                ppo_updates_num += 1
                print(f'PPO updated # {ppo_updates_num}')

            if done or timestep >= max_timesteps:
                fms.reset()
                break

        print(f'Episode {ep + 1}: Reward = {total_reward:.4f}, Steps: {timestep}')
        episode_rewards.append(total_reward)
        average_rewards.append(sum(episode_rewards[-100:]) / min(len(episode_rewards), 100))

    return episode_rewards, average_rewards


# 7. Тестирование агента
def test_agent(env_name, policy_network, render=False):
    env = gym.make(env_name)
    state, info = env.reset()
    fms.reset()
    done = False
    total_reward = 0

    # Настройки записи
    video_folder = './videos'
    os.makedirs(video_folder, exist_ok=True)

    # Тестирование агента
    test_episodes = 5  # Количество эпизодов для записи

    for i in range(test_episodes):
        video_path = os.path.join(video_folder, f'test_episode_{i}.mp4')
        recorder = VideoRecorder(env, path=video_path)

        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            recorder.capture_frame()
            state = fms(state)
            action, _, _ = select_continuous_action(state, policy_network)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        recorder.close()
        print(f'Test Episode {i + 1}: Reward = {total_reward}')


if __name__ == "__main__":
    env_name = 'CarRacing-v2'
    max_episodes = 5000
    max_timesteps = 2000
    update_timestep = 2048
    batch_size = 128
    K_epochs = 10
    gamma = 0.99
    lamda = 0.95
    eps_clip = 0.2

    # Инициализация модели политики
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # if env_name == 'CarRacing-v2':
    #     state_dim = 84
    policy_network = ContinuousPolicyNetwork(state_dim, action_dim).to('cuda')
    optimizer = optim.Adam(policy_network.parameters(), lr=1e-2)

    # Обучение агента
    episode_rewards, average_rewards = train(
        env_name,
        max_episodes,
        max_timesteps,
        update_timestep,
        batch_size,
        K_epochs,
        gamma,
        lamda,
        eps_clip,
        policy_network,
        optimizer
    )

    # Сохранение модели
    torch.save(policy_network.state_dict(), 'policy_net.pth')

    # Загрузка и тестирование агента
    policy_network.load_state_dict(torch.load('policy_net.pth'))
    test_agent(env_name, policy_network, render=True)
