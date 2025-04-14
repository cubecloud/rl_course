import sys

sys.path.insert(0, '~/Python/projects/rl_course/deeprl')

from pytz import timezone
import gymnasium as gym
from deeprl.threaded.rlagents import A2CAgent, PPOAgent
from deeprl.threaded.rlbase import RLBase

__version__ = 0.052

TZ = timezone('Europe/Moscow')

if __name__ == '__main__':
    """ Testing for CarRacing-v2 """
    from deeprl_configs.configppo_carracingv2 import ConfigAgent

    to_learn = 4000
    env_kwargs = dict(id=ConfigAgent.ENV_NAME,
                      render_mode=None)
    env = gym.make(**env_kwargs)

    rl = RLBase(env_kwargs, PPOAgent, agents_num=1, config=ConfigAgent, agents_devices=['cuda',])
    #
    rl.fit(to_learn,
           condition='episode',
           progress_bar=True,
           use_checkpoint_dir=None,
           weights_only=False,
           )
    rl.evaluate(5)
    rl.learning_curve(show_figure=False)

    # uncomment for evaluation with checkpoint weights from exp-250412-085657 and checkpoint 4000
    # rl.evaluate(5, use_checkpoint_dir='./deeprl/threaded/CarRacing-v2/PPO/exp-250412-085657/training/eps-4000')

