import argparse
from datetime import datetime
import json
import os

# Architectural hyperparameters
DEFAULT_HIDDEN_CHANNELS = 256        
DEFAULT_DROPOUT = 0.3               
DEFAULT_AGGREGATION = 'mean'	# Options: 'mean', 'sum'	
DEFAULT_NUM_GNN_LAYERS = 3           
DEFAULT_NUM_MLP_LAYERS = 3
DEFAULT_ARCHITECTURE='SageConv' #Options: 'SageConv', 'Gatv2Conv'

# Training Parameters     
DEFAULT_LEARNING_RATE = 0.005        
DEFAULT_WEIGHT_DECAY = 5e-4          
DEFAULT_NUM_EPOCHS = 300             
DEFAULT_EARLY_STOPPING_PATIENCE = 100

# Data Loading 
DEFAULT_BATCH_SIZE = 512          
DEFAULT_NUM_NEIGHBORS = [30, 20, 10] 
DEFAULT_NEG_SAMPLING_RATIO = 3.0
DEFAULT_NEG_SAMPLING = 'triplet' #Options: 'uniform', 'triplet'

# Data Split
DEFAULT_NUM_VAL = 0.10
DEFAULT_NUM_TEST = 0.10

# Learning Rate Scheduler
DEFAULT_USE_LR_SCHEDULER = True
DEFAULT_LR_SCHEDULER_FACTOR = 0.5
DEFAULT_LR_SCHEDULER_PATIENCE = 10

#Type of Loss function
DEFAULT_LOSS = 'L2' # Options: 'L2', 'L1', 'L2_weighted', 'L2_focal'
DEFAULT_LOSS_WEIGHT_TYPE = 'quadratic'  # Options: 'linear', 'quadratic', 'cubic', 'exponential'
DEFAULT_LOSS_MIN_WEIGHT = 1.0           # Minimum weight for low ratings
DEFAULT_LOSS_GAMMA = 2.0                # Focal loss gamma parameter (for focal_mse)



# Gradient Clipping
DEFAULT_USE_GRADIENT_CLIPPING = True
DEFAULT_GRAD_CLIP_VALUE = 1.0

#Embedding Regularization
DEFAULT_EMB_REG = 5e-4

# Specify which type of Knowledge Jump must be used
DEFAULT_JK_MODE = 'max'  # Options: 'cat', 'max', 'lstm'

# Specify the folder in which is stored the dataset
DEFAULT_DIRNAME_MOVIELENS = '../data/movielens/ml-latest-small'

# Specify if using Batch Normalization

DEFAULT_USE_BN = True

# Specify if using movielens TAGS as a feature of the movie or a new edge between user and movie

DEFAULT_TAG_AS_EDGE = True

#Specify modality of the software (Training or Inference)

DEFAULT_MODALITY = 'training'

#Specify where are the nn model and, if any, the local embedder

DEFAULT_MODEL_PATH = './best_model.pt'
DEFAULT_LOAD_MODEL = False
DEFAULT_EMBEDDER_PATH = '/tmp/'


def parse_arguments():
    """
    Parse command line arguments for the GNN recommendation system.
    """
    parser = argparse.ArgumentParser(
        description='Graph Neural Network for Recommendation System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    script_group = parser.add_argument_group('Script modalities')
    script_group.add_argument(
        '--modality', 
        type=str, 
        default=DEFAULT_MODALITY,
        choices=['training','inference'],
        help='Modality of the software.Training for training a model, both from scratch or finetuning and inference for using for inference task'
    )
    script_group.add_argument(
        '--model_path', 
        type=str, 
        default=DEFAULT_MODEL_PATH,
        help='Filepath of the model to be loaded or created'
    )
    
    script_group.add_argument(
        '--embedder_path', 
        type=str, 
        default=DEFAULT_EMBEDDER_PATH,
        help='Filepath of the embedder to be loaded or created'
    )
   
    # Model Architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument(
        '--hidden-channels', 
        type=int, 
        default=DEFAULT_HIDDEN_CHANNELS,
        help='Number of hidden channels in the GNN layers'
    )
    script_group.add_argument(
        '--architecture', 
        type=str, 
        default=DEFAULT_ARCHITECTURE,
        help='GNN architecture to be used'
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
        '--num-mlp-layers', 
        type=int, 
        default=DEFAULT_NUM_MLP_LAYERS,
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
        default=DEFAULT_USE_BN,
        help='Use batch normalization'
    )
    model_group.add_argument(
        '--load_model', 
        action='store_true', 
        default=DEFAULT_LOAD_MODEL,
        help='Load an already trained model'
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
        '--loss_type', 
        type=str, 
        default=DEFAULT_LOSS,
        choices=['L2','L1', 'L2_weighted', 'L2_focal'],
        help='Choosing the specific loss'
    )
    train_group.add_argument(
        '--learning-rate', '--lr',
        type=float, 
        default=DEFAULT_LEARNING_RATE,
        help='Learning rate for optimizer'
    )

    train_group.add_argument(
        '--loss-weight-type',
        type=str, 
        default=DEFAULT_LOSS_WEIGHT_TYPE,
        help='In case of L2 weighted loss, specifies how to weight'
    )

    train_group.add_argument(
        '--loss-minimum-weight',
        type=float, 
        default=DEFAULT_LOSS_MIN_WEIGHT,
        help='In case of L2 weighted loss, specifies the minimum loss weight'
    )

    train_group.add_argument(
        '--loss-gamma-param',
        type=float, 
        default=DEFAULT_LOSS_GAMMA,
        help='In case of L2 focal loss, specifies gamma parameter'
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
        '--neg-sampling', 
        type=str, 
        default=DEFAULT_NEG_SAMPLING,
        help='How to generate negative samples to positive samples'
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
        default=False,
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
        default=DEFAULT_USE_GRADIENT_CLIPPING,
        help='Use gradient clipping'
    )
    grad_group.add_argument(
        '--no-use-gradient-clipping', 
        action='store_false', 
        dest='use-gradient-clipping',
        help='Disable gradient clipping'
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
NUM_MLP_LAYERS = args.num_mlp_layers
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
NUM_EPOCHS = args.num_epochs
EARLY_STOPPING_PATIENCE = args.early_stopping_patience
BATCH_SIZE = args.batch_size
NUM_NEIGHBORS = args.num_neighbors
NEG_SAMPLING_RATIO = args.neg_sampling_ratio
NEG_SAMPLING = args.neg_sampling
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
MODALITY = args.modality
MODEL_PATH = args.model_path
EMBEDDER_PATH = args.embedder_path
LOAD_MODEL = args.load_model
ARCHITECTURE = args.architecture
LOSS_TYPE = args.loss_type
LOSS_WEIGHT_TYPE = args.loss_weight_type
LOSS_MIN_WEIGHT = args.loss_minimum_weight
LOSS_GAMMA = args.loss_gamma_param

current_params = {
    'HIDDEN_CHANNELS': args.hidden_channels,
    'DROPOUT': args.dropout,
    'AGGREGATION': args.aggregation,
    'NUM_GNN_LAYERS': args.num_gnn_layers,
    'NUM_MLP_LAYERS': args.num_mlp_layers,
    'ARCHITECTURE': args.architecture,
    'LEARNING_RATE': args.learning_rate,
    'WEIGHT_DECAY': args.weight_decay,
    'NUM_EPOCHS': args.num_epochs,
    'EARLY_STOPPING_PATIENCE': args.early_stopping_patience,
    'BATCH_SIZE': args.batch_size,
    'NUM_NEIGHBORS': args.num_neighbors,
    'NEG_SAMPLING_RATIO': args.neg_sampling_ratio,
    'NEG_SAMPLING': args.neg_sampling,
    'NUM_VAL': args.num_val,
    'NUM_TEST': args.num_test,
    'DIRNAME_MOVIELENS': args.dirname_movielens,
    'TAG_AS_EDGE': args.tag_as_edge,
    'USE_BN': args.use_bn,
    'EMB_REG': args.emb_reg,
    'JK_MODE': args.jk_mode,
    'USE_LR_SCHEDULER': args.use_lr_scheduler,
    'LR_SCHEDULER_FACTOR': args.lr_scheduler_factor,
    'LR_SCHEDULER_PATIENCE': args.lr_scheduler_patience,
    'USE_GRADIENT_CLIPPING': args.use_gradient_clipping,
    'GRAD_CLIP_VALUE': args.grad_clip_value,
    'MODALITY': args.modality,
    'MODEL_PATH': args.model_path,
    'EMBEDDER_PATH': args.embedder_path,
    'LOAD_MODEL': args.load_model,
    'LOSS_TYPE' : args.loss_type,
    'LOSS_WEIGHT_TYPE' :args.loss_weight_type,
    'LOSS_MIN_WEIGHT' : args.loss_minimum_weight,
    'LOSS_GAMMA' : args.loss_gamma_param
}

if(TAG_AS_EDGE):
    print('Since using edge that includes tag, switching the GNN architecture to Gatv2Conv')
    ARCHITECTURE = 'Gatv2Conv'
    current_params['ARCHITECTURE'] = 'Gatv2Conv'


if (MODALITY == 'inference'):
        
        model_filename = os.path.basename(MODEL_PATH)
        model_name_base = os.path.splitext(model_filename)[0]
        log_data = f"{model_name_base}_params_data.json"

        try:
            with open(log_data, 'r') as f:
                current_params = json.load(f)    
                for key, value in current_params.items():
                    globals()[key] = value
                    print(f"\n✓ Reading configuration saved in the file: {log_data}, overwriting default and input parameters")
                MODALITY = 'inference'
                current_params['MODALITY'] = 'inference'
        except IOError as e:
            print(f"Error in loading the config file {log_data}: {e}")
            print('Trying with input and default parameters')

elif (MODALITY == 'training' and LOAD_MODEL):
        
        model_filename = os.path.basename(MODEL_PATH)
        model_name_base = os.path.splitext(model_filename)[0]
        log_data = f"{model_name_base}_params_data.json"

        try:
            with open(log_data, 'r') as f:
                loaded_params = json.load(f)    
                
                current_params['NUM_MLP_LAYERS']  = loaded_params['NUM_MLP_LAYERS']
                current_params['NUM_GNN_LAYERS']  = loaded_params['NUM_GNN_LAYERS']
                current_params['ARCHITECTURE']    = loaded_params['ARCHITECTURE']
                current_params['TAG_AS_EDGE']     = loaded_params['TAG_AS_EDGE']
                current_params['EMBEDDER_PATH']   = loaded_params['EMBEDDER_PATH']
                current_params['LOSS_TYPE']       = loaded_params['LOSS_TYPE']
                current_params['HIDDEN_CHANNELS'] = loaded_params['HIDDEN_CHANNELS']
                current_params['AGGREGATION']     = loaded_params['AGGREGATION']
                
                NUM_MLP_LAYERS  = current_params['NUM_MLP_LAYERS']
                NUM_GGN_LAYERS  = current_params['NUM_GNN_LAYERS']
                ARCHITECTURE    = current_params['ARCHITECTURE']
                TAG_AS_EDGE     = current_params['TAG_AS_EDGE']
                EMBEDDER_PATH   = current_params['EMBEDDER_PATH']
                LOSS_TYPE       = current_params['LOSS_TYPE']
                HIDDEN_CHANNELS = current_params['HIDDEN_CHANNELS']
                AGGREGATION     = current_params['AGGREGATION']

        except IOError as e:
            print(f"Error in loading the config file {log_data}: {e}")
            print('Trying with input and default parameters')

# Print all parameters
print("Configuration:")
for key in current_params.keys():
    print (f"{key} -> {current_params.get(key)}")

