import argparse


#DEFAULT VALUES

DEFAULT_HIDDEN_CHANNELS = 64        # Much larger (was 96-128)
DEFAULT_DROPOUT = 0.5                # Minimal dropout (was 0.2-0.3)
DEFAULT_AGGREGATION = 'mean'
#AGGREGATION = 'sum'
DEFAULT_NUM_GNN_LAYERS = 2           # 3 LAYERS for deeper network (was 2)

# Training Parameters - AGGRESSIVE LEARNING
DEFAULT_LEARNING_RATE = 0.001        # Higher learning rate (was 0.003-0.005)
DEFAULT_WEIGHT_DECAY = 5e-4          # Minimal regularization (was 1e-4 to 2e-4)
DEFAULT_NUM_EPOCHS = 300             # More training time (was 150-200)
DEFAULT_EARLY_STOPPING_PATIENCE = 100 # Much more patience (was 20-30)

# Data Loading - MAXIMUM GRAPH INFORMATION
DEFAULT_BATCH_SIZE = 256             # Large batches (reduce to 256 if GPU OOM)
DEFAULT_NUM_NEIGHBORS = [15, 10] # 3 VALUES for 3 LAYERS! More neighbors = more info
DEFAULT_NEG_SAMPLING_RATIO = 3.0

# Data Split
DEFAULT_NUM_VAL = 0.10
DEFAULT_NUM_TEST = 0.10

# Learning Rate Scheduler
DEFAULT_USE_LR_SCHEDULER = True
DEFAULT_LR_SCHEDULER_FACTOR = 0.5
DEFAULT_LR_SCHEDULER_PATIENCE = 10   # More patience before reducing LR

# Gradient Clipping
#USE_GRADIENT_CLIPPING = True
DEFAULT_USE_GRADIENT_CLIPPING = False
DEFAULT_GRAD_CLIP_VALUE = 1.0

#Embedding Regularization
DEFAULT_EMB_REG = 5e-4

DEFAULT_JK_MODE = 'max'  # Options: 'cat', 'max', 'lstm'


DEFAULT_DIRNAME_MOVIELENS = '../data/movielens/ml-latest-small'

DEFAULT_USE_BN = True

DEFAULT_TAG_AS_EDGE = True




def parse_arguments():
    """
    Parse command line arguments for the GNN recommendation system.
    """
    parser = argparse.ArgumentParser(
        description='Graph Neural Network for Recommendation System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model Architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument(
        '--hidden-channels', 
        type=int, 
        default=DEFAULT_HIDDEN_CHANNELS,
        help='Number of hidden channels in the GNN layers'
    )
    model_group.add_argument(
        '--dropout', 
        type=float, 
        default=DEFAULT_DROPOUT,
        help='Dropout rate for regularization'
    )
    model_group.add_argument(
        '--aggregation', 
        type=str, 
        default=DEFAULT_AGGREGATION,
        choices=['mean', 'sum', 'max'],
        help='Aggregation method for neighbor features'
    )
    model_group.add_argument(
        '--num-gnn-layers', 
        type=int, 
        default=DEFAULT_NUM_GNN_LAYERS,
        help='Number of GNN layers in the model'
    )
    model_group.add_argument(
        '--jk-mode', 
        type=str, 
        default=DEFAULT_JK_MODE,
        choices=['cat', 'max', 'lstm'],
        help='Jumping knowledge mode for combining layer outputs'
    )
    model_group.add_argument(
        '--use-bn', 
        action='store_true', 
        default=True,
        help='Use batch normalization'
    )
    model_group.add_argument(
        '--no-bn', 
        action='store_false', 
        dest='use_bn',
        help='Disable batch normalization'
    )
    
    # Training Parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument(
        '--learning-rate', '--lr',
        type=float, 
        default=DEFAULT_LEARNING_RATE,
        help='Learning rate for optimizer'
    )
    train_group.add_argument(
        '--weight-decay', '--wd',
        type=float, 
        default=DEFAULT_WEIGHT_DECAY,
        help='Weight decay (L2 regularization) for optimizer'
    )
    train_group.add_argument(
        '--num-epochs', 
        type=int, 
        default=DEFAULT_NUM_EPOCHS,
        help='Maximum number of training epochs'
    )
    train_group.add_argument(
        '--early-stopping-patience', 
        type=int, 
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help='Number of epochs to wait before early stopping'
    )
    train_group.add_argument(
        '--emb-reg', 
        type=float, 
        default=DEFAULT_EMB_REG,
        help='Embedding regularization weight'
    )
    
    # Data Loading Parameters
    data_group = parser.add_argument_group('Data Loading')
    data_group.add_argument(
        '--batch-size', 
        type=int, 
        default=DEFAULT_BATCH_SIZE,
        help='Batch size for training'
    )
    data_group.add_argument(
        '--num-neighbors', 
        type=int, 
        nargs='+',
        default=DEFAULT_NUM_NEIGHBORS,
        help='Number of neighbors to sample at each layer (space-separated list)'
    )
    data_group.add_argument(
        '--neg-sampling-ratio', 
        type=float, 
        default=DEFAULT_NEG_SAMPLING_RATIO,
        help='Ratio of negative samples to positive samples'
    )
    data_group.add_argument(
        '--dirname-movielens', 
        type=str, 
        default=DEFAULT_DIRNAME_MOVIELENS,
        help='Directory path for MovieLens dataset'
    )
    data_group.add_argument(
        '--tag-as-edge', 
        action='store_true', 
        default=True,
        help='Use tags as edges in the graph'
    )
    data_group.add_argument(
        '--no-tag-as-edge', 
        action='store_false', 
        dest='tag_as_edge',
        help='Do not use tags as edges'
    )
    
    # Data Split
    split_group = parser.add_argument_group('Data Split')
    split_group.add_argument(
        '--num-val', 
        type=float, 
        default=DEFAULT_NUM_VAL,
        help='Fraction of data for validation (0.0 to 1.0)'
    )
    split_group.add_argument(
        '--num-test', 
        type=float, 
        default=DEFAULT_NUM_TEST,
        help='Fraction of data for testing (0.0 to 1.0)'
    )
    
    # Learning Rate Scheduler
    scheduler_group = parser.add_argument_group('Learning Rate Scheduler')
    scheduler_group.add_argument(
        '--use-lr-scheduler', 
        action='store_true', 
        default=True,
        help='Use learning rate scheduler'
    )
    scheduler_group.add_argument(
        '--no-lr-scheduler', 
        action='store_false', 
        dest='use_lr_scheduler',
        help='Disable learning rate scheduler'
    )
    scheduler_group.add_argument(
        '--lr-scheduler-factor', 
        type=float, 
        default=DEFAULT_LR_SCHEDULER_FACTOR,
        help='Factor by which to reduce learning rate'
    )
    scheduler_group.add_argument(
        '--lr-scheduler-patience', 
        type=int, 
        default=DEFAULT_LR_SCHEDULER_PATIENCE,
        help='Number of epochs to wait before reducing LR'
    )
    
    # Gradient Clipping
    grad_group = parser.add_argument_group('Gradient Clipping')
    grad_group.add_argument(
        '--use-gradient-clipping', 
        action='store_true', 
        default=False,
        help='Use gradient clipping'
    )
    grad_group.add_argument(
        '--grad-clip-value', 
        type=float, 
        default=DEFAULT_GRAD_CLIP_VALUE,
        help='Maximum gradient norm for clipping'
    )
    
    args = parser.parse_args()
    
    # Validation: Check that num_neighbors length matches num_gnn_layers
    if len(args.num_neighbors) != args.num_gnn_layers:
        parser.error(
            f"Length of --num-neighbors ({len(args.num_neighbors)}) must match "
            f"--num-gnn-layers ({args.num_gnn_layers})"
        )
    
    return args





args = parse_arguments()
    
# Convert args to uppercase variable names (to match your original format)
HIDDEN_CHANNELS = args.hidden_channels
DROPOUT = args.dropout
AGGREGATION = args.aggregation
NUM_GNN_LAYERS = args.num_gnn_layers
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
NUM_EPOCHS = args.num_epochs
EARLY_STOPPING_PATIENCE = args.early_stopping_patience
BATCH_SIZE = args.batch_size
NUM_NEIGHBORS = args.num_neighbors
NEG_SAMPLING_RATIO = args.neg_sampling_ratio
NUM_VAL = args.num_val
NUM_TEST = args.num_test
USE_LR_SCHEDULER = args.use_lr_scheduler
LR_SCHEDULER_FACTOR = args.lr_scheduler_factor
LR_SCHEDULER_PATIENCE = args.lr_scheduler_patience
USE_GRADIENT_CLIPPING = args.use_gradient_clipping
GRAD_CLIP_VALUE = args.grad_clip_value
EMB_REG = args.emb_reg
JK_MODE = args.jk_mode
DIRNAME_MOVIELENS = args.dirname_movielens
USE_BN = args.use_bn
TAG_AS_EDGE = args.tag_as_edge
    
# Print all parameters
print("Configuration:")
print(f"  Hidden Channels: {HIDDEN_CHANNELS}")
print(f"  Dropout: {DROPOUT}")
print(f"  Aggregation: {AGGREGATION}")
print(f"  Num GNN Layers: {NUM_GNN_LAYERS}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Weight Decay: {WEIGHT_DECAY}")
print(f"  Num Epochs: {NUM_EPOCHS}")
print(f"  Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Num Neighbors: {NUM_NEIGHBORS}")
print(f"  Negative Sampling Ratio: {NEG_SAMPLING_RATIO}")
print(f"  Validation Split: {NUM_VAL}")
print(f"  Test Split: {NUM_TEST}")
print(f"  Use LR Scheduler: {USE_LR_SCHEDULER}")
print(f"  LR Scheduler Factor: {LR_SCHEDULER_FACTOR}")
print(f"  LR Scheduler Patience: {LR_SCHEDULER_PATIENCE}")
print(f"  Use Gradient Clipping: {USE_GRADIENT_CLIPPING}")
print(f"  Gradient Clip Value: {GRAD_CLIP_VALUE}")
print(f"  Embedding Regularization: {EMB_REG}")
print(f"  JK Mode: {JK_MODE}")
print(f"  MovieLens Directory: {DIRNAME_MOVIELENS}")
print(f"  Use Batch Normalization: {USE_BN}")
print(f"  Tag as Edge: {TAG_AS_EDGE}")
