import os
import sys
import copy
import imageio
import numpy as np

from PIL import Image
from typing import Callable, Optional, Union
from collections import deque
from subprocess import run, PIPE

import shutil as sh
from glob import glob
from base64 import b64encode
import gymnasium as gym
from gymnasium.utils.save_video import save_video

__version__ = 0.029


class RewardsNormalizer:
    def __init__(self, normalizer_method: str = 'std', norm_max: float = 200.):
        self.normalizer_method = normalizer_method
        self.norm_max = norm_max
        self.abs_max: float = self.norm_max
        if self.normalizer_method == 'std':
            self.norm_method: Callable = self._std_normalizer
        elif self.normalizer_method == 'exp':
            self.norm_method: Callable = self._exp_normalizer

    def __call__(self, rewards) -> np.array:
        return self.norm_method(rewards)

    def _exp_normalizer(self, rewards) -> np.array:
        self.abs_max = np.maximum(self.abs_max, np.max(rewards))
        return 1 / (1 + np.exp(np.asarray(-rewards, dtype=np.float32) / self.abs_max)) - 0.5

    def _std_normalizer(self, rewards) -> np.array:
        return (rewards - np.mean(rewards)) / np.std(rewards)

    def reset(self):
        self.abs_max = self.norm_max


class RollingRewardNormalizer:
    def __init__(self, window: int = 100):
        self.window = window
        self.history = np.zeros(self.window)
        self.count: int = 0

    def __call__(self, reward):
        self.history[self.count] = reward
        self.count = (self.count + 1) % self.window
        return np.mean(self.history)

    def reset(self):
        self.history = np.zeros(self.window)
        self.count: int = 0


class FrameStack:
    def __init__(self, num_frames: int = 4, norm=True):
        self.num_frames = num_frames
        self.norm = norm
        self.frames = deque(maxlen=self.num_frames)

    def rgb2gray(self, rgb_frame):
        gray_frame = np.dot(rgb_frame[..., :3], [0.2989, 0.5870, 0.1140])
        if self.norm:
            gray_frame = gray_frame / 128. - 1.
        return gray_frame

    def preprocess_frame(self, frame) -> np.ndarray:
        frame = self.rgb2gray(frame).astype(np.float32)
        return np.expand_dims(frame, axis=0)

    def __call__(self, frame) -> np.ndarray:
        self.add_frame(frame)
        return self.get_stacked_frames()

    def add_frame(self, frame):
        self.frames.append(self.preprocess_frame(frame))
        while len(self.frames) < self.frames.maxlen:
            self.frames.append(self.frames[0])

    def get_stacked_frames(self) -> np.ndarray:
        return np.concatenate(self.frames, axis=0)

    def reset(self):
        self.frames.clear()


def scale_car_actions(action):
    # rescaling for box [-1, 1 ][0, 1][0, 1] actions
    action = action - np.array([0, -1., -1.]) / np.array([1., 2., 2.])
    return action


class EnvFrameStackedWrapper:
    """
    Environment wrapper for stack frames and calculating reward
    for example CarRacing-v2
    """

    def __init__(self, env: Optional = None, env_kwargs: Optional[dict] = None, stack_frames: int = 4,
                 action_repeat: int = 8, reward_norm_window: int = 100, ma_reward_condition: float = -0.1,
                 color_control: Optional[float] = None, scale_action: Callable = None):
        self.initial_kwargs: dict = {}
        self.save_initial_kwargs(env=env, env_kwargs=env_kwargs, stack_frames=stack_frames,
                                 action_repeat=action_repeat, reward_norm_window=reward_norm_window,
                                 ma_reward_condition=ma_reward_condition, color_control=color_control,
                                 scale_action=scale_action)
        if env is not None:
            self.env = env
            self.env_kwargs = env_kwargs
            self.render_mode = None
        else:
            assert env_kwargs is not None, f'"env_kwargs" and "env" is empty, use at least one of this kwargs'
            self.env_kwargs = env_kwargs
            self.render_mode = env_kwargs.get('render_mode')
            self.env = gym.make(**env_kwargs)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.spec = self.env.spec
        self.stack_frames = stack_frames
        self.action_repeat = action_repeat
        self.ma_reward_condition = ma_reward_condition

        if scale_action is not None:
            self.scale_action: Callable = scale_action
        else:
            self.scale_action: scale_action = lambda x: x

        if color_control is not None:
            self.color_control = color_control
            self.color_check: Callable = self._color_check_reward
        else:
            self.color_control = color_control
            self.color_check: Callable = lambda x: x[0]

        self.fms = FrameStack(num_frames=stack_frames)
        self.ma_rrn = RollingRewardNormalizer(window=reward_norm_window)
        self.render_frames = deque(maxlen=self.action_repeat)

    def _color_check_reward(self, img_rgb, current_reward) -> Union[float, int]:
        if np.mean(img_rgb[:, :, 1]) > self.color_control:
            current_reward -= 0.05
        return current_reward

    def step(self, action):
        total_reward = 0
        img_rgb = None
        done = False
        terminated = False
        info = {}
        action = self.scale_action(action)
        for i in range(self.action_repeat):
            img_rgb, reward, terminated, truncated, info = self.env.step(action)
            # don't penalize "die state"
            # if terminated:
            #     reward += 100
            # penalty for prevail color
            reward = self._color_check_reward(img_rgb, reward)
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.ma_rrn(reward) <= self.ma_reward_condition else False
            if self.render_mode is not None:
                self.render_frames.append(self.env.render())
            if done or terminated:
                break
        return self.fms(img_rgb), total_reward, done, terminated, info

    def render(self):
        return self.render_frames

    def reset(self):
        self.fms.reset()
        self.render_frames.clear()
        img_rgb, info = self.env.reset()
        if self.render_mode is not None:
            self.render_frames.append(self.env.render())
        return self.fms(img_rgb), info

    def save_initial_kwargs(self, **kwargs):
        self.initial_kwargs = dict(**kwargs)


def save_movie_gif(eps_frames, gif_path_filename: str):
    saveanimation(eps_frames, gif_path_filename)


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


def save_mp4(eps_frames, path_filename, fps=25):
    eps_frame_dir = 'episode_frames'

    # Create a temporary directory
    try:
        os.makedirs(eps_frame_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {e}")
        return

    # Save each frame as a PNG file
    for i, frame in enumerate(eps_frames):
        try:
            image_file = os.path.join(eps_frame_dir, f'frame-{i + 1:04d}.png')
            Image.fromarray(frame).save(image_file)  # Save the frame as PNG
        except Exception as e:
            print(f"Error saving frame {i}: {e}")
            continue  # Continue processing other frames

    # Run ffmpeg command to convert images into an MP4 video
    command = [
        'ffmpeg', '-v', '0', '-r', str(fps),
        '-i', os.path.join(eps_frame_dir, 'frame-%04d.png'),
        '-vcodec', 'libx264', '-crf', '18', '-y', path_filename
    ]

    try:
        result = run(command, stdout=PIPE, stderr=PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
    except Exception as e:
        print(f"Error running ffmpeg: {e}")

    # Remove the temporary directory
    try:
        if os.path.exists(eps_frame_dir):
            sh.rmtree(eps_frame_dir)
    except OSError as e:
        print(f"Error deleting directory: {e}")


def show_records(records_path):
    record_paths = glob(os.path.join(records_path, "*.mp4"))
    html_str = ''
    for i, record_path in enumerate(record_paths):
        mp4 = open(record_path, 'rb').read()
        data = f"data:video/mp4;base64,{b64encode(mp4).decode()}"
        html_str += f'EPISODE # {i + 1}<br><video width=500 controls><source src="{data}" type="video/mp4"></video><br><br>'
    return html_str


def show_mp4(path_filename):
    mp4 = open(path_filename, 'rb').read()
    data = f"data:video/mp4;base64,{b64encode(mp4).decode()}"
    html_str = f'<br><video width=500 controls><source src="{data}" type="video/mp4"></video><br><br>'
    return html_str


def play_and_save_mp4(agent, num_episodes: int = 3, truncate_steps: int = 1000, video_folder: str = "videos",
                      name_prefix: str = "rl_video", fps: int = 25):
    new_env_kwargs = copy.deepcopy(agent.env_kwargs)
    new_env_kwargs.update({'render_mode': 'rgb_array_list'})
    env = gym.make(**new_env_kwargs)

    agent.env = env
    episode_reward = 0.
    stdout_save = sys.stdout
    z = open(os.devnull, 'w')
    for episode_index in range(num_episodes):
        state, info = env.reset()
        for step_index in range(truncate_steps):
            action = agent.get_action(state, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            done = terminated or truncated
            if done:
                # stdout off
                sys.stdout = z
                save_video(
                    frames=env.render(),
                    video_folder=video_folder,
                    fps=fps,
                    episode_trigger=lambda x: True,
                    episode_index=episode_index,
                    name_prefix=name_prefix,
                )
                break
    # stdout on
    sys.stdout = stdout_save
    env = gym.make(**agent.env_kwargs)
    agent.env = env
