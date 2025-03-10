import os
import sys
sys.path.insert(0, '~/Python/projects/rl_course/deeprl')

import time
import logging
import datetime
from pytz import timezone
import numpy as np
import gymnasium as gym

from threading import Thread

from typing import List, Union

from tqdm import tqdm

# from rlcollection.configdqn_lunarlanderv2 import ConfigAgent
# from rlcollection.configa2c_lunarlanderv2 import ConfigAgent
# from rlcollection.configa2c_cartpolev1 import ConfigAgent
from deeprl.rlsync import RLSYNC_obj
from deeprl.rlagents import DQNAgent
from deeprl.rlagents import A2CAgent
from deeprl.rlagents import show_mp4, show_records
from deeprl.rlbase import RLDQN

import matplotlib.pyplot as plt
import pandas as pd

__version__ = 0.030

TZ = timezone('Europe/Moscow')

logger = logging.getLogger()


if __name__ == '__main__':
    """ Testing for Taxi-v3 """
    from deeprl_configs.configdqn_taxiv3 import ConfigAgent

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(f'{ConfigAgent.ENV_NAME}.log')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL.Image').setLevel(logging.WARNING)
    """----------------------------------------------------------------"""

    truncate_steps = 200
    to_learn = 10000
    env = gym.make(ConfigAgent.ENV_NAME,
                   render_mode='rgb_array'
                   )

    rl = RLDQN(env, DQNAgent, config=ConfigAgent, agents_num=2, agents_devices=['cuda', 'cpu'])
    rl.fit(to_learn,
           condition='episode',
           progress_bar=True,
           use_checkpoint_dir=None,
           weights_only=False,
           )
    rl.evaluate(5)
    rl.learning_curve(show_figure=True)
