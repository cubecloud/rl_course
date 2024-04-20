class ConfiqDQN:
    BUFFER_SIZE = 30_000    # replay buffer size
    BATCH_SIZE = 512        # batch size
    GAMMA = 0.95            # discount factor
    SYNC_FRAME = 19          # how often to update the target_network (frames)
    EPS_START = 0.99
    EPS_END = 0.05
    EPS_DECAY = 1_000
    TAU = 0.005
    LR = 1e-4
