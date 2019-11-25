import torch

# Random
CUDA_AVAILABLE = True and torch.cuda.is_available()
USE_TENSORBOARD = True
TF_PATH = 'tensorboard'
EXPERIMENT_PATH_PREFIX = 'experiments/cvae'
# Below is per iteration
LOG_INTERVAL = 50

# Below are per epoch
TEST_INTERVAL = 100
SAVE_INTERVAL = 10000

# Training params
INITIAL_LEARNING_RATE = 0.0001
LEARNING_RATE_UPDATE_CYCLES = 15000
HIDDEN_LAYERS = [400, 400, 400]
WEIGHT_DECAY = 0.0001
REPLAY_BUFFER_MAX_SIZE = 50000
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
# Num samples to take while drawing samples during test
TEST_SAMPLES = 1000

# Env params
X_MAX = 20
Y_MAX = 15
POINT_DIM = 2

# State params
X_DIM = POINT_DIM
# Start/goal, base/arm encoding, walls
C_DIM = 2 * POINT_DIM + 11 * POINT_DIM
# C_DIM = 2 * POINT_DIM + POINT_DIM #+ 11 * POINT_DIM
LATENT_DIM = 10





