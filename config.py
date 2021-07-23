import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEG_BATCH_SIZE = 16
VID_BATCH_SIZE = 7
LEARNING_RATE = 1e-4
NUM_EPOCHS = 300
LOAD_MODEL = False

SYNPICK_CLASSES = 22

VID_DATA_ALLOW_OVERLAP = False
VIDEO_IN_LENGTH = 6
VIDEO_PRED_LENGTH = 3
VIDEO_TOT_LENGTH = VIDEO_IN_LENGTH + VIDEO_PRED_LENGTH