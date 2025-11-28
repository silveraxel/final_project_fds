import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import JumpingKnowledge

# ============================================================================
# HYPERPARAMETERS - OPTIMIZED FOR MAXIMUM PERFORMANCE
# ============================================================================
"""
SITUATION: Good generalization but poor performance
DIAGNOSIS: Model is UNDERFITTING
SOLUTION: Increase capacity + reduce regularization + train more aggressively
"""

# Model Architecture - MAXIMUM CAPACITY
HIDDEN_CHANNELS = 256        # Much larger (was 96-128)
DROPOUT = 0.5                # Minimal dropout (was 0.2-0.3)
#AGGREGATION = 'mean'
AGGREGATION = 'sum'
NUM_GNN_LAYERS = 3           # 3 LAYERS for deeper network (was 2)

# Training Parameters - AGGRESSIVE LEARNING
LEARNING_RATE = 0.01        # Higher learning rate (was 0.003-0.005)
WEIGHT_DECAY = 1e-4          # Minimal regularization (was 1e-4 to 2e-4)
NUM_EPOCHS = 300             # More training time (was 150-200)
EARLY_STOPPING_PATIENCE = 100 # Much more patience (was 20-30)

# Data Loading - MAXIMUM GRAPH INFORMATION
BATCH_SIZE = 256             # Large batches (reduce to 256 if GPU OOM)
NUM_NEIGHBORS = [40, 20, 10] # 3 VALUES for 3 LAYERS! More neighbors = more info
NEG_SAMPLING_RATIO = 2.0

# Data Split
NUM_VAL = 0.20
NUM_TEST = 0.10

# Learning Rate Scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 15   # More patience before reducing LR

# Gradient Clipping
#USE_GRADIENT_CLIPPING = True
USE_GRADIENT_CLIPPING = False
GRAD_CLIP_VALUE = 1.0


JK_MODE = 'cat'  # Options: 'cat', 'max', 'lstm'

# ============================================================================


DIRNAME_MOVIELENS = '../data/movielens/ml-latest-small'

movie_path = os.path.join(DIRNAME_MOVIELENS,'movies.csv')
rating_path = os.path.join(DIRNAME_MOVIELENS,'ratings.csv')
tags_path = os.path.join(DIRNAME_MOVIELENS,'tags.csv')

def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping



def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, one_edge_per_row=False, **kwargs):
    """
    Unified edge loading function for both ratings and tags.
    
    Parameters:
    -----------
    path : str
        Path to the CSV file
    src_index_col : str
        Column name for source nodes (e.g., 'userId')
    src_mapping : dict
        Mapping from source node IDs to indices
    dst_index_col : str
        Column name for destination nodes (e.g., 'movieId')
    dst_mapping : dict
        Mapping from destination node IDs to indices
    encoders : dict, optional
        Dictionary mapping column names to encoder objects
    one_edge_per_row : bool, default=False
        If True, create one edge per row (for tags where duplicates are meaningful)
        If False, assume unique (src, dst) pairs (for ratings)
    **kwargs : dict
        Additional arguments passed to pd.read_csv()
    
    Returns:
    --------
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges]
    edge_attr : torch.Tensor or None
        Edge attributes (if encoders provided)
    """
    df = pd.read_csv(path, **kwargs)
    
    if one_edge_per_row:
        # Tags mode: Each row becomes one edge (handles duplicates)
        edge_src = []
        edge_dst = []
        edge_attrs = []
        
        # Apply encoders first if provided
        encoded_features = None
        if encoders is not None:
            # Handle special case for tag encoders that return (embeddings, df)
            for col, encoder in encoders.items():
                result = encoder(df)
                if isinstance(result, tuple):  # TagEncoder returns (embeddings, df)
                    encoded_features, df = result
                else:  # Standard encoder returns just embeddings
                    encoded_features = result
        
        # Create edges row by row
        for idx, row in df.iterrows():
            src_id = row[src_index_col]
            dst_id = row[dst_index_col]
            
            # Only add edge if both nodes exist in mappings
            if src_id in src_mapping and dst_id in dst_mapping:
                edge_src.append(src_mapping[src_id])
                edge_dst.append(dst_mapping[dst_id])
                
                if encoded_features is not None:
                    edge_attrs.append(encoded_features[idx])
        
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.stack(edge_attrs) if edge_attrs else None
        
    else:
        # Ratings mode: Assume unique (src, dst) pairs (original behavior)
        src = [src_mapping[index] for index in df[src_index_col]]
        dst = [dst_mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst])
        
        edge_attr = None
        if encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)
    
    return edge_index, edge_attr



class SequenceEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class GenresEncoder:
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = {g for col in df.values for g in col.split(self.sep)}
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x


class IdentityEncoder:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)



class TagEncoder:
    """Encode individual tags as embeddings (no aggregation)"""
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        """Encode each individual tag (no aggregation or artificial sentences)"""
        # Encode each unique tag
        unique_tags = df['tag'].unique()
        print(f"Encoding {len(unique_tags)} unique tags...")
        
        tag_embeddings = self.model.encode(
            unique_tags,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device
        ).cpu()
        
        # Create mapping
        tag_to_embedding = {tag: tag_embeddings[i] for i, tag in enumerate(unique_tags)}
        
        # Return embeddings for each row
        embeddings = []
        for _, row in df.iterrows():
            embeddings.append(tag_to_embedding[row['tag']])
        
        embeddings = torch.stack(embeddings)
        
        # Return (embeddings, df) tuple for compatibility
        return embeddings, df


# ============================================================================
# IMPROVED MODEL WITH 3 GNN LAYERS
# ============================================================================

class GNN(torch.nn.Module):
    """GNN with Jumping Knowledge connections and batch normalization"""
    def __init__(self, hidden_channels, dropout=0.0, use_bn=False, num_layers=3, jk_mode='cat'):
        super().__init__()
        self.num_layers = num_layers
        self.jk_mode = jk_mode
        
        # Convolutional layers (dynamically created based on num_layers)
        self.convs = torch.nn.ModuleList([
            SAGEConv((-1, -1), hidden_channels) for _ in range(num_layers)
        ])
        
        self.dropout = torch.nn.Dropout(dropout)
        
        # Batch normalization
        self.use_bn = use_bn
        if use_bn:
            self.bns = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
            ])
        
        # Jumping Knowledge connection
        # Modes: 'cat' (concatenation), 'max' (max pooling), 'lstm' (LSTM aggregation)
        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=num_layers)
        
        # Output projection (only needed for 'cat' mode)
        if jk_mode == 'cat':
            self.output_proj = torch.nn.Linear(hidden_channels * num_layers, hidden_channels)
        
    def forward(self, x, edge_index):
        # Store outputs from each layer
        layer_outputs = []
        
        # Apply all convolutional layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            if self.use_bn:
                x = self.bns[i](x)
            
            x = x.relu()
            x = self.dropout(x)
            
            # Store this layer's output for JK
            layer_outputs.append(x)
        
        # Apply Jumping Knowledge to combine all layer outputs
        x = self.jk(layer_outputs)
        
        # Project back to hidden_channels if using concatenation
        if self.jk_mode == 'cat':
            x = self.output_proj(x)
        
        return x

class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, dropout=0.0):
        super().__init__()
        
        input_dim = hidden_channels * 2
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.5),
            
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.5),
            
            torch.nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, x_user, x_movie, edge_label_index):
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        combined = torch.cat([edge_feat_user, edge_feat_movie], dim=-1)
        return self.mlp(combined).squeeze(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, metadata, dropout=0.0, use_bn=False, 
                 num_gnn_layers=3, jk_mode='cat'):
        super().__init__()
        
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        
        # GNN with Jumping Knowledge
        self.gnn = GNN(
            hidden_channels, 
            dropout=dropout, 
            use_bn=use_bn, 
            num_layers=num_gnn_layers,
            jk_mode=jk_mode  # Pass JK mode
        )
        self.gnn = to_hetero(self.gnn, metadata, aggr=AGGREGATION)
        
        self.classifier = Classifier(hidden_channels, dropout=dropout * 0.5)

    def forward(self, data):
        user_emb = self.user_emb(data['user'].n_id)
        
        x_dict = {
            'user': user_emb,
            'movie': data['movie'].x,
        }
        
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        pred = self.classifier(
            x_dict['user'],
            x_dict['movie'],
            data['user', 'rates', 'movie'].edge_label_index
        )
        return pred

# ============================================================================
# LOAD DATA
# ============================================================================

user_x, user_mapping = load_node_csv(rating_path, index_col='userId')

movie_x, movie_mapping = load_node_csv(
    movie_path, index_col='movieId', encoders={
        'title': SequenceEncoder(),
        'genres': GenresEncoder()
    })


edge_index, edge_label = load_edge_csv(
    rating_path,
    src_index_col='userId',
    src_mapping=user_mapping,
    dst_index_col='movieId',
    dst_mapping=movie_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)},
    one_edge_per_row=False  
)




# ============================================================================
# NEW: Load tags and create TAG EDGES
# ============================================================================


print("\nLoading tags data and creating tag edges...")
try:
    tag_edge_index, tag_edge_attr = load_edge_csv(
        tags_path,
        src_index_col='userId',
        src_mapping=user_mapping,
        dst_index_col='movieId',
        dst_mapping=movie_mapping,
        encoders={'tag': TagEncoder()},
        one_edge_per_row=True
    )
    
    print(f"Created {tag_edge_index.size(1)} tag edges")
    print(f"Tag edge features shape: {tag_edge_attr.shape}")
    
    tags_enabled = True
    
except FileNotFoundError:
    print("Warning: tags.csv not found, proceeding without tag edges")
    tag_edge_index = None
    tag_edge_attr = None
    tags_enabled = False
except Exception as e:
    print(f"Warning: Error loading tags: {e}, proceeding without tag edges")
    tag_edge_index = None
    tag_edge_attr = None
    tags_enabled = False


# ============================================================================
# CREATE HETEROGENEOUS GRAPH WITH TAG EDGES
# ============================================================================

data = HeteroData()

# Nodes
data['user'].num_nodes = len(user_mapping)
data['movie'].num_nodes = len(movie_mapping)
data['movie'].x = movie_x

# Rating edges (user --[rates]--> movie)
data['user', 'rates', 'movie'].edge_index = edge_index
data['user', 'rates', 'movie'].edge_label = edge_label

# Tag edges (user --[tags]--> movie)

data['user', 'tags', 'movie'].edge_index = tag_edge_index
data['user', 'tags', 'movie'].edge_attr = tag_edge_attr
print(f"\n✓ Added 'tags' edge type to the graph")

print("\nHeterogeneous graph structure:")
print(data)

# Add reverse edges
data = ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label
del data['movie', 'rev_tags', 'user'].edge_label

# Split data
transform = RandomLinkSplit(
    num_val=NUM_VAL,
    num_test=NUM_TEST,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')],
)
train_data, val_data, test_data = transform(data)

print("\nTrain data:", train_data)
print("Val data:", val_data)
print("Test data:", test_data)

# ============================================================================
# SETUP DEVICE AND MODEL
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

num_users = len(user_mapping)

model = Model(
    hidden_channels=HIDDEN_CHANNELS,
    num_users=num_users,
    metadata=train_data.metadata(),
    dropout=DROPOUT,
    use_bn=True,
    num_gnn_layers=NUM_GNN_LAYERS,
    jk_mode='cat'
).to(device)

print(f"\nModel Configuration:")
print(f"  Hidden Channels: {HIDDEN_CHANNELS}")
print(f"  GNN Layers: {NUM_GNN_LAYERS}")
print(f"  Dropout: {DROPOUT}")
print(f"  Batch Normalization: Enabled")

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Learning rate scheduler
if USE_LR_SCHEDULER:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=LR_SCHEDULER_FACTOR, 
        patience=LR_SCHEDULER_PATIENCE,
        verbose=True
    )

# ============================================================================
# CREATE DATA LOADERS
# ============================================================================

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=NUM_NEIGHBORS,
    neg_sampling_ratio=NEG_SAMPLING_RATIO,
    edge_label_index=(('user', 'rates', 'movie'), 
                      train_data['user', 'rates', 'movie'].edge_label_index),
    edge_label=train_data['user', 'rates', 'movie'].edge_label,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=NUM_NEIGHBORS,
    edge_label_index=(('user', 'rates', 'movie'),
                      val_data['user', 'rates', 'movie'].edge_label_index),
    edge_label=val_data['user', 'rates', 'movie'].edge_label,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=NUM_NEIGHBORS,
    edge_label_index=(('user', 'rates', 'movie'),
                      test_data['user', 'rates', 'movie'].edge_label_index),
    edge_label=test_data['user', 'rates', 'movie'].edge_label,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train():
    model.train()
    total_loss = 0
    total_examples = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch)
        
        ground_truth = batch['user', 'rates', 'movie'].edge_label.squeeze().float()
        
        loss = F.mse_loss(pred, ground_truth)
        
        loss.backward()
        
        if USE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
        
        optimizer.step()
        
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()
    total_loss = 0
    total_examples = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        pred = model(batch)
        ground_truth = batch['user', 'rates', 'movie'].edge_label.squeeze().float()
        
        loss = F.mse_loss(pred, ground_truth)
        
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    
    rmse = (total_loss / total_examples) ** 0.5
    return rmse

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("TRAINING WITH TAG EDGES")
print("="*80)
print(f"Hyperparameters:")
print(f"  Hidden Channels: {HIDDEN_CHANNELS}")
print(f"  GNN Layers: {NUM_GNN_LAYERS}")
print(f"  Dropout: {DROPOUT}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Weight Decay: {WEIGHT_DECAY}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Num Neighbors: {NUM_NEIGHBORS}")
print(f"  Neg Sampling Ratio: {NEG_SAMPLING_RATIO}")
print(f"  Max Epochs: {NUM_EPOCHS}")
print("="*80 + "\n")

best_val_rmse = float('inf')
best_train_rmse = float('inf')
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    loss = train()
    train_rmse = test(train_loader)
    val_rmse = test(val_loader)
    
    train_val_gap = val_rmse - train_rmse
    
    if USE_LR_SCHEDULER:
        scheduler.step(val_rmse)
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_train_rmse = train_rmse
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
        improvement_marker = "✓"
    else:
        patience_counter += 1
        improvement_marker = ""
    
    
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train: {train_rmse:.4f}, Val: {val_rmse:.4f}, '
          f'Gap: {train_val_gap:.4f}')
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f'\n✓ Early stopping at epoch {epoch}')
        break

# Load best model
model.load_state_dict(torch.load('best_model.pt'))



# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

movies_df = pd.read_csv(movie_path)
reverse_user_mapping = {v: k for k, v in user_mapping.items()}
reverse_movie_mapping = {v: k for k, v in movie_mapping.items()}


@torch.no_grad()
def predict_rating(user_id, movie_id):
    """Predict rating for a specific user-movie pair"""
    model.eval()
    
    if user_id not in user_mapping or movie_id not in movie_mapping:
        return None
    
    user_idx = user_mapping[user_id]
    movie_idx = movie_mapping[movie_id]
    
    edge_label_index = torch.tensor([[user_idx], [movie_idx]], device=device)
    
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=NUM_NEIGHBORS,
        edge_label_index=(('user', 'rates', 'movie'), edge_label_index),
        batch_size=1,
        shuffle=False,
    )
    
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        return pred.item()


@torch.no_grad()
def evaluate_sample_predictions(n_users=10, n_samples_per_user=5):
    """Sample n_users, predict n_samples per user, and plot prediction errors"""
    model.eval()
    
    print(f"\n{'='*80}")
    print(f"SAMPLE PREDICTIONS - {n_users} Users x {n_samples_per_user} Movies Each")
    print(f"{'='*80}\n")
    
    test_edges = test_data['user', 'rates', 'movie'].edge_label_index
    test_labels = test_data['user', 'rates', 'movie'].edge_label
    
    # Group test edges by user
    user_edges = {}
    for i in range(test_edges.size(1)):
        user_idx = test_edges[0, i].item()
        if user_idx not in user_edges:
            user_edges[user_idx] = []
        user_edges[user_idx].append(i)
    
    eligible_users = [u for u, edges in user_edges.items() if len(edges) >= n_samples_per_user]
    if len(eligible_users) < n_users:
        print(f"Warning: Only {len(eligible_users)} users have {n_samples_per_user}+ ratings")
        n_users = len(eligible_users)
    
    sampled_users = np.random.choice(eligible_users, size=n_users, replace=False)
    
    all_errors = []
    all_actual = []
    all_predicted = []
    user_labels = []
    
    for user_idx in sampled_users:
        user_id = reverse_user_mapping[user_idx]
        
        available_edges = user_edges[user_idx]
        sampled_edge_indices = np.random.choice(available_edges, size=n_samples_per_user, replace=False)
        
        print(f"\nUser {user_id}:")
        user_errors = []
        
        for edge_idx in sampled_edge_indices:
            movie_idx = test_edges[1, edge_idx].item()
            actual_rating = test_labels[edge_idx].item()
            
            movie_id = reverse_movie_mapping[movie_idx]
            movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
            
            predicted_rating = predict_rating(user_id, movie_id)
            
            error = abs(predicted_rating - actual_rating)
            user_errors.append(error)
            all_errors.append(error)
            all_actual.append(actual_rating)
            all_predicted.append(predicted_rating)
            
            print(f"  {movie_title[:50]:50s} | Actual: {actual_rating:.1f} | Pred: {predicted_rating:.2f} | Error: {error:.2f}")
        
        avg_user_error = np.mean(user_errors)
        print(f"  → Average Error for User {user_id}: {avg_user_error:.3f}")
        user_labels.extend([f"User {user_id}"] * n_samples_per_user)
    
    all_errors = np.array(all_errors)
    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total Predictions: {len(all_errors)}")
    print(f"Mean Absolute Error (MAE): {np.mean(all_errors):.3f}")
    print(f"Std of Errors: {np.std(all_errors):.3f}")
    print(f"Min Error: {np.min(all_errors):.3f}")
    print(f"Max Error: {np.max(all_errors):.3f}")
    print(f"RMSE: {np.sqrt(np.mean(all_errors**2)):.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Prediction Error Analysis ({n_users} Users × {n_samples_per_user} Predictions)', 
                 fontsize=14, fontweight='bold')
    
    axes[0, 0].hist(all_errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_errors):.3f}')
    axes[0, 0].set_xlabel('Absolute Error', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(all_actual, all_predicted, alpha=0.6, s=50, color='coral')
    axes[0, 1].plot([0, 5], [0, 5], 'k--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('Actual Rating', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Rating', fontsize=11)
    axes[0, 1].set_title('Actual vs Predicted Ratings', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 5.5)
    axes[0, 1].set_ylim(0, 5.5)
    
    user_error_data = []
    user_names = []
    for user_idx in sampled_users:
        user_id = reverse_user_mapping[user_idx]
        user_mask = np.array([label == f"User {user_id}" for label in user_labels])
        user_error_data.append(all_errors[user_mask])
        user_names.append(f"U{user_id}")
    
    bp = axes[1, 0].boxplot(user_error_data, labels=user_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    axes[1, 0].set_xlabel('User ID', fontsize=11)
    axes[1, 0].set_ylabel('Absolute Error', fontsize=11)
    axes[1, 0].set_title('Error Distribution per User', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].scatter(all_actual, all_errors, alpha=0.6, s=50, color='mediumpurple')
    axes[1, 1].axhline(np.mean(all_errors), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(all_errors):.3f}')
    axes[1, 1].set_xlabel('Actual Rating', fontsize=11)
    axes[1, 1].set_ylabel('Absolute Error', fontsize=11)
    axes[1, 1].set_title('Error vs Actual Rating', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 5.5)
    
    plt.tight_layout()
    
    plot_filename = 'prediction_error_analysis.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved as: {plot_filename}")
    
    plt.show()
    
    print(f"{'='*80}\n")
    
    return all_errors, all_actual, all_predicted


# Run sample predictions with visualization
errors, actual, predicted = evaluate_sample_predictions(n_users=10, n_samples_per_user=5)