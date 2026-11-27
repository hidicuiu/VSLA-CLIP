from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Optional stage-1 checkpoint used by train_reidadapter.py. A non-empty
# --stage1weight command-line argument still takes precedence.
_C.MODEL.STAGE1_WEIGHT = ''
# Opt-in validation for paired runs that must share the complete initial model.
_C.MODEL.STAGE1_STRICT_LOAD = False

# M3-ReID-inspired frame aggregation; mean preserves the original E4 path.
_C.MODEL.FRAME_AGGREGATION = 'mean'  # mean / multihead / uniform_fusion / single_shared / conditional*
_C.MODEL.FRAME_AGGREGATION_HEADS = 2
_C.MODEL.FRAME_AGGREGATION_GROUPS = 1  # channel groups per head; 1 keeps legacy checkpoints
_C.MODEL.FRAME_AGGREGATION_HIDDEN_DIM = 64
_C.MODEL.FRAME_AGGREGATION_DIVERSITY_WEIGHT = 0.0
# Opt-in primary/conditional-supplement pooling and matched controls.
_C.MODEL.FRAME_AGGREGATION_HEAD_LOSS_WEIGHT = 0.0
_C.MODEL.FRAME_AGGREGATION_SUPPRESSION = 0.5
_C.MODEL.FRAME_AGGREGATION_GATE_INIT = 0.1

# Training-only two-head set contrast; zero preserves the existing baseline.
_C.MODEL.CROSS_PLATFORM_SET_WEIGHT = 0.0
_C.MODEL.CROSS_PLATFORM_SET_TEMPERATURE = 0.1
_C.MODEL.CROSS_PLATFORM_SET_MATCHING = 'optimal'  # optimal / fixed / fused (post-fusion SupCon)
_C.MODEL.CROSS_PLATFORM_SET_CAMERA_TO_PLATFORM = []

_C.MODEL.FUSION_HEAD_NUM = 12
_C.MODEL.FUSION_LAYER_NUM = 2
_C.MODEL.ATT_TYPE = ['c2c', 'o2o']  # ['c2c', 'o2o', 'c2o', 'o2c']
_C.MODEL.ADAPTER_TYPE = 'dat'
_C.MODEL.ADAPTER_ALPHA = 4

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS_WEIGHT = 1.0
_C.MODEL.T2I_WEIGHT = 1.0
_C.MODEL.I2T_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# PBP Parameter
_C.MODEL.PBP_COE = 3.0
_C.MODEL.PBP_CAMERA = False
_C.MODEL.PBP_VIEW = False

# ViFi
_C.MODEL.VIFI_WEIGHT = ''
_C.MODEL.USE_VIFI_WEIGHT = False
_C.MODEL.PROMPT_LEN = 8

# prompt
_C.MODEL.USE_LEARNABLE_PROMPT = True

# adapter
_C.MODEL.USE_ADAPTER = False
_C.MODEL.USE_ADAPTER_LEN = 4
# Ablation switches for the two components of the Video Set-Level Adapter.
# They only take effect when USE_ADAPTER is True.
_C.MODEL.USE_IFA = True
_C.MODEL.USE_CFAA = True
_C.MODEL.PBP_PROMPT_LEN = 1
_C.MODEL.PBP_PROMPT_DEEP = 9

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
# AG-VPReID uses the actual filename camera labels to validate the direction.
_C.DATASETS.AGVPREID_DIRECTION = 'g2a'
_C.DATASETS.AGVPREID_SPLIT_DIR = ''


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
_C.DATALOADER.SEQ_LEN = 4
_C.DATALOADER.DFGS_GRAPH = ''
_C.DATALOADER.DFGS_PROBABILITY = 1.0
_C.DATALOADER.DYNAMIC_DFGS_ENABLED = False
_C.DATALOADER.DYNAMIC_DFGS_UPDATE_PERIOD = 10
_C.DATALOADER.DYNAMIC_DFGS_SKIP = 2
_C.DATALOADER.DYNAMIC_DFGS_NEIGHBORS = 10
_C.DATALOADER.DYNAMIC_DFGS_QUALITY_WEIGHTED = False
_C.DATALOADER.DYNAMIC_DFGS_RELIABLE_NEGATIVES = False
_C.DATALOADER.DYNAMIC_DFGS_STRATIFIED_TWO_CAMERA = False
_C.DATALOADER.DYNAMIC_DFGS_STRATIFIED_MULTICAMERA = False
# Replace reliability-selected exclusions with a deterministic random set of
# the same size for a matched filtering control.
_C.DATALOADER.DYNAMIC_DFGS_RANDOM_FILTER = False
# When set, random filtering uses the paired E4 graph's exclusion count at
# each refresh epoch. The random identities are still sampled from this run.
_C.DATALOADER.DYNAMIC_DFGS_RANDOM_FILTER_REFERENCE_DIR = ''
# Match the reference filter's per-anchor candidate impact and rank/similarity
# distributions while keeping the sampled IDs disjoint from the reference IDs.
_C.DATALOADER.DYNAMIC_DFGS_RANDOM_FILTER_MATCH_CANDIDATE_IMPACT = False
_C.DATALOADER.DYNAMIC_DFGS_RANDOM_FILTER_MATCH_ITERATIONS = 6000
_C.DATALOADER.DYNAMIC_DFGS_RANDOM_FILTER_MATCH_RESTARTS = 4
# Optional two-platform reliability filtering; camera IDs still feed PBP/eval.
_C.DATALOADER.DYNAMIC_DFGS_STRATIFIED_PLATFORM = False
# Entry i gives the platform of camera i: 0=ground, 1=aerial.
_C.DATALOADER.DYNAMIC_DFGS_CAMERA_TO_PLATFORM = []
_C.DATALOADER.DYNAMIC_DFGS_MIN_OBSERVED_CAMERAS = 2
_C.DATALOADER.DYNAMIC_DFGS_MIN_LOW_CAMERA_FRACTION = 0.5
# Reliability thresholds may be stratified independently by camera and/or
# tracklet length when multicamera filtering is enabled. Defaults preserve E4.
_C.DATALOADER.DYNAMIC_DFGS_GROUP_BY_CAMERA = True
_C.DATALOADER.DYNAMIC_DFGS_GROUP_BY_LENGTH = True
_C.DATALOADER.DYNAMIC_DFGS_CANDIDATE_POOL = 30
_C.DATALOADER.DYNAMIC_DFGS_LOW_QUALITY_FRACTION = 0.20
_C.DATALOADER.DYNAMIC_DFGS_SOFT_NEGATIVES = False
_C.DATALOADER.DYNAMIC_DFGS_LOW_CANDIDATE_WEIGHT = 0.25
_C.DATALOADER.DYNAMIC_DFGS_BATCH_SIZE = 8
_C.DATALOADER.DYNAMIC_DFGS_NUM_WORKERS = 8
# native: use the training/retrieval aggregation; mean: bypass only the new
# final aggregation for graph centers AND both temporal reliability views.
_C.DATALOADER.DYNAMIC_DFGS_READOUT = 'native'  # native / mean
_C.DATALOADER.DYNAMIC_DFGS_SAVE_GRAPHS = True
# Keep graph extraction from consuming the training RNG streams in paired runs.
_C.DATALOADER.DYNAMIC_DFGS_PRESERVE_RNG_STATE = False
_C.DATALOADER.PID_SAMPLES_PER_EPOCH = 0
_C.DATALOADER.RHP_GRAPH = ''

# ---------------------------------------------------------------------------- #
# Solver
_C.SOLVER = CN()
_C.SOLVER.SEED = 1234
_C.SOLVER.MARGIN = 0.3

# stage1
# ---------------------------------------------------------------------------- #
# Name of optimizer
_C.SOLVER.STAGE1 = CN()

_C.SOLVER.STAGE1.IMS_PER_BATCH = 64

_C.SOLVER.STAGE1.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.STAGE1.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.STAGE1.BASE_LR = 3e-4
# Momentum
_C.SOLVER.STAGE1.MOMENTUM = 0.9

# Settings of weight decay
_C.SOLVER.STAGE1.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE1.WEIGHT_DECAY_BIAS = 0.0005

# warm up factor
_C.SOLVER.STAGE1.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.STAGE1.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE1.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE1.LR_MIN = 0.000016

_C.SOLVER.STAGE1.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.STAGE1.WARMUP_METHOD = "linear"

_C.SOLVER.STAGE1.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE1.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.STAGE1.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.STAGE1.LOG_PERIOD = 100
# epoch number of validation
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
# _C.SOLVER.STAGE1.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1.EVAL_PERIOD = 10

# ---------------------------------------------------------------------------- #
# Solver
# stage1
# ---------------------------------------------------------------------------- #
_C.SOLVER.STAGE2 = CN()

_C.SOLVER.STAGE2.IMS_PER_BATCH = 64
# Name of optimizer
_C.SOLVER.STAGE2.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.STAGE2.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.STAGE2.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.STAGE2.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.STAGE2.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.STAGE2.MOMENTUM = 0.9
# Margin of triplet loss
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.STAGE2.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.STAGE2.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.STAGE2.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE2.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.STAGE2.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STAGE2.STEPS = (40, 70)
# warm up factor
_C.SOLVER.STAGE2.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.STAGE2.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE2.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE2.LR_MIN = 0.000016


_C.SOLVER.STAGE2.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.STAGE2.WARMUP_METHOD = "linear"

_C.SOLVER.STAGE2.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE2.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.STAGE2.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.STAGE2.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.STAGE2.EVAL_PERIOD = 10
# When true, only the final epoch is evaluated and no test-selected best
# checkpoint is written. The final epoch checkpoint is always saved.
_C.SOLVER.STAGE2.FINAL_CHECKPOINT_ONLY = False
# Reset all training seeds at the Stage-2 boundary, independently of Stage 1.
# Also disables cuDNN benchmarking.
_C.SOLVER.STAGE2.RESET_SEED = False
# Refuse to overwrite prior runs independently of the random-state protocol.
_C.SOLVER.STAGE2.PROTECT_OUTPUT = False
# Evaluation adapters for completed runs must never launch another training job.
_C.SOLVER.STAGE2.EVALUATION_ONLY = False
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# AG-VPReID standalone test directions; empty uses DATASETS.AGVPREID_DIRECTION.
# Training checkpoint selection still uses DATASETS.AGVPREID_DIRECTION only.
_C.TEST.AGVPREID_DIRECTIONS = []
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
_C.TEST.ALL_FRAMES = False
_C.TEST.DISTANCE = 'cosine'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
_C.TEST.SEQ_LEN = 4
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
