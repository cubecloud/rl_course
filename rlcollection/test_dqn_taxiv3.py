import os
import sys
sys.path.insert(0, '/home/cubecloud/Python/projects/rl_course/rlcollection')

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
from rlcollection.rlsync import RLSYNC_obj
from rlcollection.rlagents import DQNAgent
from rlcollection.rlagents import A2CAgent
from rlcollection.rlagents import show_mp4, show_records
from rlcollection.rlbase import RLDQN

import matplotlib.pyplot as plt
import pandas as pd

__version__ = 0.030

TZ = timezone('Europe/Moscow')

logger = logging.getLogger()


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('check_lock.log')
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
    """ Testing for Taxi-v3 """
    from configdqn_taxiv3 import ConfigAgent

    to_learn = 60000
    env = gym.make("Taxi-v3",
                   render_mode='rgb_array'
                   )
    # env = gym.make("CartPole-v1",
    #                render_mode='rgb_array',
    #                )

    rl = RLDQN(env, DQNAgent, agents_num=2, agents_devices=['cuda', 'cpu'], config=ConfigAgent)
    # rl = RLDQN(env, DQNAgent, agents_num=2, agents_devices=['cuda', 'cpu', ], config=ConfigAgent)
    # rl = RLDQN(env, DQNAgent, agents_num=1, agents_devices=['cpu'], config=ConfigAgent)
    # rl = RLA2C(env, A2CAgent, agents_num=1, agents_devices=['cuda'], config=ConfigAgent)
    # rl = RLA2C(env, A2CAgent, agents_num=1, agents_devices=['cuda'], config=ConfigAgent)
    # rl = RLA2C(env, A2CAgent, agents_num=2, agents_devices=['cuda', 'cpu'], config=ConfigAgent)
    rl.fit(to_learn,
           condition='episode',
           progress_bar=True,
           use_checkpoint_dir=None,
           weights_only=False,
           )
    rl.evaluate(5)
    rl.learning_curve(show_figure=True)
