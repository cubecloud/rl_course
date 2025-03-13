class ConfigAgent:
    ENV_NAME = "CartPole-v1"
    BUFFER_SIZE = 10_000         # replay buffer size
    BATCH_SIZE = 256             # batch size
    MASKED_ACTION = False
    GAMMA = 0.99                 # discount factor
    SYNC_FRAME = 3              # how often to update the target_network (frames) or learn every X from buffer episode for A2C
    AGENTS_SYNC_FRAME = 25      # how often to update the networks between running agents (frames) (must be > 10)
    EPS_START = 0.00
    EPS_END = 0.00
    EPS_DECAY = 1_000_000
    TAU = 1e-4
    LR = 1e-3
    EXPERIMENT_PATH = "./deeprl/threaded/"
    REWARD_CONDITION = 200
    VALIDATION_STEP = 500
    VALIDATION_RATIO = 0.1
    VALIDATION_EPS = 10
    EVALUATION_EPS = 5
