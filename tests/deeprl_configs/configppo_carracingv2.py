class ConfigAgent:
    ENV_NAME = "CarRacing-v2"
    BUFFER_SIZE = 3000           # replay buffer size
    BATCH_SIZE = 12               # batch size (episodes)
    MASKED_ACTION = False
    GAMMA = 0.99                 # discount factor
    SYNC = 12                     # how often we use replay buffer (episodes) to update target network
    AGENTS_SYNC = 12              # how often to update the networks (weights) between running agents (episodes)
    EPS_START = 0.00
    EPS_END = 0.00
    EPS_DECAY = 1_000_000
    TAU = 3e-4                  # updating agents networks with this rate
    LR = 1e-3                   # learning rate
    EXPERIMENT_PATH = "./deeprl/threaded/"
    REWARD_CONDITION = 500      # reward condition to check in validation for (win_ratio calculation)
    VALIDATION_STEP = 500        # validate every n episode
    VALIDATION_RATIO = 0.1      # validation ratio * validation step = episodes to validate
    VALIDATION_EPS = 5          # validation episodes to save video
    EVALUATION_EPS = 5          # evaluation episodes to save video
