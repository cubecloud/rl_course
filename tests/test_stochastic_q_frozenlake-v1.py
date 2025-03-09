import gymnasium as gym
from gymnasium.utils.save_video import save_video
from tabularrl.tabularstochastic import StochasticModel, TabularStochasticQAgent


if __name__ == "__main__":
    env_id = "FrozenLake-v1"
    truncate_steps = 100
    """
    for generating new map use folowing code:
        from gymnasium.envs.toy_text.frozen_lake import generate_random_map
        desc = generate_random_map(size=6)
        print(desc)
        # add printed 'desc' below for using it for learning
        desc = ['SFFFHF', 'FFFFFF', 'FFHHHF', 'HFFFFH', 'FFFFHF', 'FFFFFG']
    """
    desc = ['SFFHF', 'FFFFF', 'FHFHF', 'FFFHF', 'FFFFG']
    map_name = f'{len(desc)}x{len(desc)}'
    env_kwargs = dict(id=env_id,
                      desc=desc,
                      is_slippery=True,
                      map_name=map_name
                      )
    agent_kwargs = dict(learning_rate=0.1, gamma=0.99)
    train_kwargs = dict(episodes=15000, truncate_steps=truncate_steps, tb_log='TB')
    print('Use: tensorboard --logdir "path to project"/rl_course/tests/TB')

    real_env = gym.make(**env_kwargs)
    virtual_env = StochasticModel(real_env)
    agent = TabularStochasticQAgent(virtual_env, **agent_kwargs)
    agent.train(**train_kwargs)

    # Loading agent and evaluate learned agent
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
    print('Check "videos" directory in "path to project"/rl_course/tests/videos')
