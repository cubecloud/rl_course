class ConfigAgent:
    ENV_NAME = "CartPole-v1"
    BUFFER_SIZE = 1500          # replay buffer size (frames)
    BATCH_SIZE = 1              # batch size (episodes)
    MASKED_ACTION = False
    GAMMA = 0.99                # discount factor
    SYNC = 2                    # how often we use replay buffer (episodes) to update target network
    AGENTS_SYNC = 5             # how often to update the networks (weights) between running agents (episodes)
    EPS_START = 0.00
    EPS_END = 0.00
    EPS_DECAY = 1_000_000
    TAU = 5e-4                  # updating agents networks with this rate
    LR = 5e-4                   # learning rate
    EXPERIMENT_PATH = "./deeprl/threaded/"
    REWARD_CONDITION = 500      # reward condition to check in validation for (win_ratio calculation)
    VALIDATION_STEP = 1000      # validate every n episode
    VALIDATION_RATIO = 0.1      # validation ratio * validation step = episodes to validate
    VALIDATION_EPS = 10         # validation episodes to save video
    EVALUATION_EPS = 5          # evaluation episodes to save video
