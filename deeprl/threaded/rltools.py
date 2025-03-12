import os
import sys
import copy
import imageio
from subprocess import run, PIPE
from PIL import Image

import shutil as sh
from glob import glob
from base64 import b64encode
import gymnasium as gym
from gymnasium.utils.save_video import save_video

__version__ = 0.011


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


def validate_episodes(agent, num_episodes: int = 3, truncate_steps: int = 1000):
    episodes_steps, episodes_rewards, episodes_dones = [], [], []
    for episodes in range(num_episodes):
        eps_steps, eps_reward, eps_done = validate_episode(agent, truncate_steps=truncate_steps)
        episodes_steps.append(eps_steps)
        episodes_rewards.append(eps_reward)
        episodes_dones.append(eps_done)
    return episodes_steps, episodes_rewards, episodes_dones


def validate_episode(agent, truncate_steps: int = 1000):
    eps_steps, eps_reward = 0, 0.
    eps_done = False
    state, info = agent.env.reset()
    for eps_steps in range(truncate_steps):
        action = agent.get_action(state, info)
        next_state, reward, terminated, truncated, info = agent.env.step(action)
        eps_reward += reward
        state = next_state
        eps_done = terminated or truncated
        if eps_done:
            break
    return eps_steps, eps_reward, eps_done
