import os
import gymnasium as gym
from gymnasium.utils.save_video import save_video
from tabularrl.tabularstochastic import StochasticModel, TabularStochasticQAgent


if __name__ == "__main__":
    env_id = "Taxi-v3"
    truncate_steps = 200
    env_kwargs = dict(id=env_id)
    exp_path = "./tabularrl/"
    agent_kwargs = dict(learning_rate=0.05, gamma=0.99, exp_path=exp_path)
    train_kwargs = dict(episodes=10000, truncate_steps=truncate_steps, tb_log='TB')
    print(f'Use: tensorboard --logdir "path to project"/rl_course/tests/tabularrl/TB')

    real_env = gym.make(**env_kwargs)
    virtual_env = StochasticModel(real_env)
    agent = TabularStochasticQAgent(virtual_env, **agent_kwargs)
    agent.train(**train_kwargs)

    # Loading agent and evaluate learned agent
    loaded_agent = TabularStochasticQAgent.load_agent(os.path.join(exp_path, f'{env_id}_agent.pkl'))

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
                    video_folder=f'{exp_path}/videos',
                    fps=env.metadata["render_fps"],
                    episode_trigger=lambda x: True,
                    episode_index=episode_index,
                    name_prefix=f'{env_id}_rl_video'
                )
                break
    env.close()
    print('Check "videos" directory in "path to project"/rl_course/tests/tabularrl/videos')
