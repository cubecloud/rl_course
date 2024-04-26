class ConfiqDQN:
    BUFFER_SIZE = 100_000        # replay buffer size
    BATCH_SIZE = 512            # batch size
    GAMMA = 0.95                # discount factor
    SYNC_FRAME = 19              # how often to update the target_network (frames)
    AGENTS_SYNC_FRAME = 100     # how often to update the networks between running agents (frames) (must be > 10)
    EPS_START = 0.95
    EPS_END = 0.05
    EPS_DECAY = 10_000
    TAU = 0.005
    LR = 1e-4
