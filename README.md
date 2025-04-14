# rl_course
### Homework for RL

Create conda environment: ```conda env create --file "path to project"/rl_course/rl-base.yml```

### Tabular RL with Q learning with Taxi-v3 and FrozenLake-v1 
1. Run: [tabularrl_frozenlake-v1.py](demo/tabularrl_frozenlake-v1.py) or [tabularrl_taxi-v3.py](demo/tabularrl_taxi-v3.py)
1. 2. Run: ```tensorboard --logdir "path to project"/rl_course/demo/tabularrl/TB'```, you can current training results 
3. Check "videos" directory in "path to project"/rl_course/demo/tabularrl/videos

### Deep RL with multithreaded agents with soft update between all agents with LunarLander-v2
1. run [deeprl_threaded_dqn_lunarlanderv2.py](demo/deeprl_threaded_dqn_lunarlanderv2.py) WARNING! long training cycle ~3h
2. or comment training cycle and uncomment evaluation with checkpoint weights

### Deep RL with PPO agent with CarRacing-v2 (unfortunately not working in threaded mode)
1. run [deeprl_threaded_ppo_carracingv2.py](demo/deeprl_threaded_ppo_carracingv2.py) WARNING! long training cycle ~3h
2. or comment training cycle and uncomment evaluation with checkpoint weights
