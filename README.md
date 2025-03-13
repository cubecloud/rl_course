# rl_course
### Homework for RL

Create conda environment: ```conda env create --file "path to project"/rl_course/rl-base.yml```

### Tabular RL with Q learning with Taxi-v3 and FrozenLake-v1 
1. Run: [test_stochastic_q_frozenlake-v1.py](tests/test_stochastic_q_frozenlake-v1.py) or [test_stochastic_q_taxi-v3.py](tests/test_stochastic_q_taxi-v3.py)
2. Run: ```tensorboard --logdir "path to project"/rl_course/tests/tabularrl/TB'```, you can current training results 
3. Check "videos" directory in "path to project"/rl_course/tests/tabularrl/videos

### Deep RL with multithreaded agents with soft update between all agents with LunarLander-v2
1. run [test_deeprl_threaded_dqn_lunarlanderv2.py](tests/test_deeprl_threaded_dqn_lunarlanderv2.py) WARNING! long training cycle ~3h
2. or comment training cycle and uncomment evaluation with checkpoint weights
