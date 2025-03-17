class ConfigAgent:
    ENV_NAME = "LunarLander-v2"
    BUFFER_SIZE = 16000         # replay buffer size
    BATCH_SIZE = 7              # batch size (episodes)
    MASKED_ACTION = False
    GAMMA = 0.99                # discount factor
    SYNC = 7                   # how often we use replay buffer (episodes) to update target network
    AGENTS_SYNC = 7             # how often to update the networks (weights) between running agents (episodes)
    EPS_START = 0.00
    EPS_END = 0.00
    EPS_DECAY = 1_000_000
    TAU = 2e-4                  # updating agents networks with this rate
    LR = 5e-5                   # learning rate
    EXPERIMENT_PATH = "./deeprl/threaded/"
    REWARD_CONDITION = 200      # reward condition to check in validation for (win_ratio calculation)
    VALIDATION_STEP = 2000      # validate every n episode
    VALIDATION_RATIO = 0.1      # validation ratio * validation step = episodes to validate
    VALIDATION_EPS = 5          # validation episodes to save video
    EVALUATION_EPS = 5          # evaluation episodes to save video
