##INSTALLATION

Install torch version compatible with CUDA
The tested version (corresponding to the available GPU) used CUDA 12.1 (actually 12.2 but same driver as 12.1)
'pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121'

After that, use 'pip -r requirements.txt'

## USE OF THE SCRIPT
Example of use of the main script

python main.py --hidden-channels 128 --num-gnn-layers 3 --num-neighbors 40 20 10 
In case you want to load an already trained model provide the flag --model-path $MODEL_PATH 
e.g. python main.pt --model_path prova.pt


## DEFAULT PARAMETERS
Here follows the parameters that can be override and their default values. If not specified, the default values is used

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
DEFAULT_LOSS = 'L2' # Options: 'L2', 'L1'

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