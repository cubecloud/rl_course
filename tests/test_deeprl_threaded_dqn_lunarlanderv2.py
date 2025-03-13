import sys

sys.path.insert(0, '~/Python/projects/rl_course/deeprl')

from pytz import timezone
import gymnasium as gym
from deeprl.threaded.rlagents import DQNAgent
from deeprl.threaded.rlbase import RLDQN

__version__ = 0.044

TZ = timezone('Europe/Moscow')

if __name__ == '__main__':
    """ Testing for LunarLander-v2 """
    from deeprl_configs.configdqn_lunarlanderv2 import ConfigAgent

    to_learn = 12000
    env_kwargs = dict(id=ConfigAgent.ENV_NAME, render_mode=None)
    env = gym.make(**env_kwargs)

    rl = RLDQN(env_kwargs, DQNAgent, agents_num=3, config=ConfigAgent, agents_devices=['cuda', 'cpu', 'cpu'])

    rl.fit(to_learn,
           condition='episode',
           progress_bar=True,
           use_checkpoint_dir=None,
           weights_only=False,
           )
    rl.evaluate(5)
    rl.learning_curve(show_figure=False)

    # uncomment for evaluation with checkpoint weights from exp-250312-172602 and checkpoint 12000
    # rl.evaluate(5, use_checkpoint_dir='./deeprl/threaded/LunarLander-v2/DQN/exp-250312-172602/training/eps-12000')
