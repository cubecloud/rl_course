import os
import datetime
import numpy as np
import pickle
import random
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from pytz import timezone

TZ = timezone('Europe/Moscow')

__version__ = 0.013


def get_exp_id() -> str:
    return f'exp-{datetime.datetime.now(TZ).strftime("%y%m%d-%H%M%S")}'


class StochasticModel:
    def __init__(self, env):
        self.env = env
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.transition_counts = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.int32)

    def step(self, state, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.update_transition_model(state, action, next_state)
        return next_state, reward, terminated, truncated, info

    def reset(self):
        return self.env.reset()

    def update_transition_model(self, state, action, next_state):
        self.transition_counts[state, action, next_state] += 1

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)

    def save_virtual_environment(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.__getstate__(), file)

    @classmethod
    def load_virtual_environment(cls, filename, real_env):
        with open(filename, 'rb') as file:
            state_dict = pickle.load(file)
        ve = cls(real_env)
        ve.__setstate__(state_dict)
        return ve


class TabularStochasticQAgent:
    def __init__(self, virtual_env, learning_rate=0.01, gamma=0.9, epsilon_start=1.0, epsilon_min=0.05, exp_path='./'):
        self.virtual_env = virtual_env
        self.env = self.virtual_env.env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.exp_path = exp_path
        self.num_states = virtual_env.num_states
        self.num_actions = virtual_env.num_actions

        # Prepare the tables
        self.Q_table = np.zeros((self.num_states, self.num_actions, self.num_states))  # Q(s, a, s')
        self.N_table = self.virtual_env.transition_counts

    def _calculate_weighted_q_value(self, q_matrix, transition_matrix):
        """
        Calculate max weighted q_values for each action based on probability and reward

        Args:
            q_matrix (np.array):            q_matrix
            transition_matrix (np.array):   n_matrix - counted transitions
        Returns:
            weighted_q_values(np.array):    weighted q_values for each action
        """
        # transitions sums for each action (transition_matrix have (num_actions, num_next_states) dimensions)

        total_transitions = np.sum(transition_matrix, axis=1)

        # Preparing data for safe divide on it
        safe_total_transitions = np.where(total_transitions == 0, 1e-8, total_transitions)

        # Transitions probability for each next_state with safe divide
        transition_probs = np.divide(transition_matrix,
                                     safe_total_transitions.reshape(-1, 1),
                                     where=(safe_total_transitions.reshape(-1, 1) != 0)
                                     )

        # Masks zero values
        masked_q_matrix = np.ma.masked_where(transition_probs == 0, q_matrix)

        # get max value for each action from masked array
        weighted_q_values = np.ma.max(masked_q_matrix, axis=1)

        return weighted_q_values

    def select_action(self, state, info):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_action(state)

    def get_action(self, state):
        # Calculate max weighted q_values for each action based on probability and reward
        weighted_q_values = self._calculate_weighted_q_value(self.Q_table[state], self.N_table[state])

        # Sum all transitions counts for all (actions, next_states) from this state
        total_transitions_sum = np.sum(self.N_table[state])

        # Probability transitions for each _action_ with safe divide
        action_weights = np.divide(np.sum(self.N_table[state], axis=1),
                                   total_transitions_sum,
                                   where=(total_transitions_sum != 0))

        # Calculate Q-values with action weights
        weighted_q_values_with_action_weights = weighted_q_values * action_weights

        # if all Q-values with action weights is zero - just use random sample from real_env
        # else
        # get max Q_value action
        if not np.any(weighted_q_values_with_action_weights):
            return self.env.action_space.sample()
        else:
            return np.argmax(weighted_q_values_with_action_weights)

    def update_Q_value(self, state, action, reward, next_state, done):
        # Getting the max next_state Q_value
        if not done:
            future_q = np.max(self.Q_table[next_state])
        else:
            future_q = 0.0

        target = reward + self.gamma * future_q

        # Update q_value
        current_q = self.Q_table[state, action, next_state]
        update_q = (self.learning_rate * (target - current_q))
        # backpropagation 1 step but for all possible next_states
        # we filter not used/ low used next_states with probability at select_action time
        self.Q_table[state, action] += update_q

    def train(self, episodes=10000, truncate_steps=200, stats=100, tb_log='TB'):
        real_env_id = self.env.spec.id
        log_dir = os.path.join(self.exp_path, tb_log, f'{real_env_id}_{get_exp_id()}')
        writer = SummaryWriter(log_dir=log_dir)  # init tensorboard SummaryWriter

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

            # Update epsilon
            self.epsilon = max(self.epsilon_min,
                               self.epsilon_start - (self.epsilon_start - self.epsilon_min) * (
                                       (episode + 1) / episodes))

            rewards.append(total_reward)
            # save best weights
            if total_reward >= max_reward:
                max_reward = total_reward
                self.save_agent(os.path.join(self.exp_path, f'{real_env_id}_agent.pkl'))

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
        return self.__dict__

    def __setstate__(self, state_dict):

        self.__dict__.update(state_dict)

    def save_agent(self, filename):
        """ Save the agent """
        with open(filename, 'wb') as file:
            pickle.dump(self.__getstate__(), file)

    @classmethod
    def load_agent(cls, filename):
        """Loading saved agent """
        with open(filename, 'rb') as file:
            state_dict = pickle.load(file)
        agent = cls(state_dict['virtual_env'])
        agent.__setstate__(state_dict)
        return agent
