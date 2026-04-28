"""
Hyperparameters 
"""

# ---------- Data ----------
IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 64

# CelebA attributes 
SELECTED_ATTRS = [
    "Smiling",
    "Male",
    "Eyeglasses",
    "Blond_Hair",
    "Young",
    "Mustache",
    "Pale_Skin",
    "Heavy_Makeup",
    "Bald",
    "Bangs",
]
NUM_CLASSES = len(SELECTED_ATTRS)  # 10

# ---------- Diffusion ----------
TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
SCHEDULE = "cosine"  # "linear" or "cosine"

# ---------- Model ----------
BASE_CHANNELS = 64
CHANNEL_MULTS = (1, 2, 4, 8)
TIME_DIM = 256
ATTR_DIM = 128

# ---------- Training ----------
LEARNING_RATE = 2e-4
EPOCHS = 100
CFG_DROP_PROB = 0.1       # classifier-free guidance dropout
SAVE_EVERY = 10           # save checkpoint every N epochs
SAMPLE_EVERY = 10         # generate samples every N epochs
NUM_SAMPLES = 16          # how many images to generate for visualization

# ---------- Sampling ----------
DDIM_STEPS = 50           # number of steps for DDIM sampling
CFG_SCALE = 5.0           # classifier-free guidance scale

# ---------- Paths ----------
CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"
