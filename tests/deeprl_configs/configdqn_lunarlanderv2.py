class ConfigAgent:
    ENV_NAME = "LunarLander-v2"
    BUFFER_SIZE = 20_000       # replay buffer size
    BATCH_SIZE = 512            # batch size
    MASKED_ACTION = False
    GAMMA = 0.99                # discount factor
    SYNC = 5              # how often to update the target_network (frames)
    AGENTS_SYNC = 19      # how often to update the networks between running agents (frames) (must be > 10)
    EPS_START = 1.00
    EPS_END = 0.05
    EPS_DECAY = 1_000_000
    TAU = 1e-3
    LR = 7e-5
    EXPERIMENT_PATH = "./deeprl/threaded/"
    REWARD_CONDITION = 200
    VALIDATION_STEP = 1000
    VALIDATION_RATIO = 0.1
    VALIDATION_EPS = 10
    EVALUATION_EPS = 5
