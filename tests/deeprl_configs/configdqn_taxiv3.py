class ConfigAgent:
    ENV_NAME = "Taxi-v3"
    BUFFER_SIZE = 10_000         # replay buffer size
    BATCH_SIZE = 1_000           # batch size
    MASKED_ACTION = True
    GAMMA = 0.99                # discount factor
    SYNC = 1000                 # how often to update the target_network (frames)
    AGENTS_SYNC = 2000          # how often to update the networks between running agents (frames) (must be > 10)
    EPS_START = 0.99
    EPS_END = 0.03
    EPS_DECAY = 1_000_000
    TAU = 5e-3
    LR = 1e-4
    EXPERIMENT_PATH = "./tests"
    REWARD_CONDITION = 10
    VALIDATION_STEP = 100
    VALIDATION_RATIO = 0.1
    VALIDATION_EPS = 10
    EVALUATION_EPS = 3

