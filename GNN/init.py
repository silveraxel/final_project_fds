from libraries import *

movie_path = os.path.join(DIRNAME_MOVIELENS,'movies.csv')
rating_path = os.path.join(DIRNAME_MOVIELENS,'ratings.csv')
tags_path = os.path.join(DIRNAME_MOVIELENS,'tags.csv')

user_x, user_mapping = load_node_csv(rating_path, index_col='userId')

if TAGS == 'edge':
    print("\nLoading tags data and creating tag edges...")
    movie_x, movie_mapping = load_node_csv(
        movie_path, index_col='movieId', encoders={
            'title': SequenceEncoder(local_path=EMBEDDER_PATH),
            'genres': GenresEncoder()
        })


    edge_index, edge_label = load_edge_csv_tags(
        rating_path,
        src_index_col='userId',
        src_mapping=user_mapping,
        dst_index_col='movieId',
        dst_mapping=movie_mapping,
        encoders={'rating': IdentityEncoder(dtype=torch.long)},
        one_edge_per_row=False  
    )

    
    try:
        
# Load tag edges (user --[tags]--> movie)
        tag_edge_index, tag_edge_attr = load_edge_csv_tags(
            tags_path,
            src_index_col='userId',
            src_mapping=user_mapping,
            dst_index_col='movieId',
            dst_mapping=movie_mapping,
            encoders={'tag': TagEncoder(hidden_channels=HIDDEN_CHANNELS)},
            one_edge_per_row=True # Each row (tag) is a new edge
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


elif TAGS == 'feature':

# First get movie mapping (needed for tag encoder)
    movies_df = pd.read_csv(movie_path)
    movies_df = movies_df.set_index('movieId')
    movie_mapping = {index: i for i, index in enumerate(movies_df.index.unique())}



    # Load movie features WITH aggregated tags
    print("\nLoading movie features with aggregated tags...")
    movie_x, _ = load_node_csv(
        movie_path, index_col='movieId', encoders={
            'title': SequenceEncoder(local_path=EMBEDDER_PATH),
            'genres': GenresEncoder(),
            'tags': AggregatedTagEncoder(tags_path, movie_mapping, local_path=EMBEDDER_PATH)  # NEW: Tags as movie features
        })

    edge_index, edge_label = load_edge_csv(
        rating_path,
        src_index_col='userId',
        src_mapping=user_mapping,
        dst_index_col='movieId',
        dst_mapping=movie_mapping,
        encoders={'rating': IdentityEncoder(dtype=torch.long)},
    )
elif TAGS == 'None':
    # Load movie features WITHOUT tags
    print("\nLoading movie features without tags...")
    movie_x, movie_mapping = load_node_csv(
        movie_path, index_col='movieId', encoders={
            'title': SequenceEncoder(local_path=EMBEDDER_PATH),
            'genres': GenresEncoder()
        })

    edge_index, edge_label = load_edge_csv(
        rating_path,
        src_index_col='userId',
        src_mapping=user_mapping,
        dst_index_col='movieId',
        dst_mapping=movie_mapping,
        encoders={'rating': IdentityEncoder(dtype=torch.long)},
    )    
else:
    print('Something went wrong, not defined the use of tags file. Exiting the program')
    exit(1)

data = HeteroData()

# Nodes
data['user'].num_nodes = len(user_mapping)
data['movie'].num_nodes = len(movie_mapping)
data['movie'].x = movie_x

# Rating edges (user --[rates]--> movie)
data['user', 'rates', 'movie'].edge_index = edge_index
data['user', 'rates', 'movie'].edge_label = edge_label

# Tag edges (user --[tags]--> movie)
if TAGS == 'edge':
    data['user', 'tags', 'movie'].edge_index = tag_edge_index
    data['user', 'tags', 'movie'].edge_attr = tag_edge_attr
    print(f"\n✓ Added 'tags' edge type to the graph")

print("\nHeterogeneous graph structure:")
print(data)

# Add reverse edges
data = ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label
if TAGS == 'edge':
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
    use_bn=USE_BN,
    num_gnn_layers=NUM_GNN_LAYERS,
    num_mlp_layers=NUM_MLP_LAYERS,
    jk_mode=JK_MODE,
    architecture=ARCHITECTURE
).to(device)


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
    neg_sampling = NEG_SAMPLING,
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




def train():
    model.train()
    total_loss = 0
    total_examples = 0

    #Loss definition
    if LOSS_TYPE == 'L2':
        criterion = torch.nn.MSELoss()
    elif LOSS_TYPE == 'L1':
        criterion = torch.nn.L1Loss()
    elif LOSS_TYPE == 'L2_weighted':
        criterion = WeightedMSELoss(weight_type=LOSS_WEIGHT_TYPE,min_weight=LOSS_MIN_WEIGHT,rating_scale=5.0)
    elif LOSS_TYPE == 'L2_focal':
        criterion = FocalMSELoss(
        rating_weight=LOSS_WEIGHT_TYPE,
        gamma=LOSS_GAMMA,
        rating_scale=5.0)
    else:
        print('Not defined a loss function. Exiting the script')
        exit(1)

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch)
        
        ground_truth = batch['user', 'rates', 'movie'].edge_label.squeeze().float()
        
        loss = criterion(pred,ground_truth)
        
        emb_reg = EMB_REG * model.user_emb.weight.norm(2)
        
        loss = loss + emb_reg

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

    #Loss definition
    if LOSS_TYPE == 'L2':
        criterion = torch.nn.MSELoss()
    elif LOSS_TYPE == 'L1':
        criterion = torch.nn.L1Loss()
    elif LOSS_TYPE == 'L2_weighted':
        criterion = WeightedMSELoss(weight_type=LOSS_WEIGHT_TYPE,min_weight=LOSS_MIN_WEIGHT,rating_scale=5.0)
    elif LOSS_TYPE == 'L2_focal':
        criterion = FocalMSELoss(
        rating_weight=LOSS_WEIGHT_TYPE,
        gamma=LOSS_GAMMA,
        rating_scale=5.0)
    else:
        print('Not defined a loss function. Exiting the script')
        exit(1)

    for batch in loader:
        batch = batch.to(device)
        
        pred = model(batch)
        ground_truth = batch['user', 'rates', 'movie'].edge_label.squeeze().float()
        
        loss = criterion(pred,ground_truth)
        
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    
    rmse = (total_loss / total_examples) ** 0.5
    return rmse

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
    

def evaluate_sample_predictions(params: dict, movies_df, n_users=10, n_samples_per_user=5, rating_threshold=3.5):
    
    model_path = params.get('MODEL_PATH', './default_model.pt')
    model_filename = os.path.basename(model_path)
    model_name_base = os.path.splitext(model_filename)[0]
    plot_filename = f"{model_name_base}_analysis.png" 

    reverse_user_mapping = {v: k for k, v in user_mapping.items()}
    reverse_movie_mapping = {v: k for k, v in movie_mapping.items()}
    
    model.eval()
    print(f"\n{'='*80}")
    print(f"SAMPLE PREDICTIONS - {n_users} Users x {n_samples_per_user} Movies Each")
    print(f"Rating threshold for binary classification: {rating_threshold}")
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
    all_actual_binary = [] 
    all_predicted_binary = []  
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
            
            # NEW: Binary classification
            actual_binary = 1 if actual_rating > rating_threshold else 0
            predicted_binary = 1 if predicted_rating > rating_threshold else 0
            all_actual_binary.append(actual_binary)
            all_predicted_binary.append(predicted_binary)
            
            print(f"  {movie_title[:50]:50s} | Actual: {actual_rating:.1f} | Pred: {predicted_rating:.2f} | Error: {error:.2f}")
        
        avg_user_error = np.mean(user_errors)
        print(f"  → Average Error for User {user_id}: {avg_user_error:.3f}")
        user_labels.extend([f"User {user_id}"] * n_samples_per_user)
    
    all_errors = np.array(all_errors)
    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    all_actual_binary = np.array(all_actual_binary)
    all_predicted_binary = np.array(all_predicted_binary)

        
    accuracy = accuracy_score(all_actual_binary, all_predicted_binary)
    precision = precision_score(all_actual_binary, all_predicted_binary, zero_division=0)
    recall = recall_score(all_actual_binary, all_predicted_binary, zero_division=0)
    f1 = f1_score(all_actual_binary, all_predicted_binary, zero_division=0)
    conf_matrix = confusion_matrix(all_actual_binary, all_predicted_binary)
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total Predictions: {len(all_errors)}")
    print(f"Mean Absolute Error (MAE): {np.mean(all_errors):.3f}")
    print(f"Std of Errors: {np.std(all_errors):.3f}")
    print(f"Min Error: {np.min(all_errors):.3f}")
    print(f"Max Error: {np.max(all_errors):.3f}")
    print(f"RMSE: {np.sqrt(np.mean(all_errors**2)):.3f}")
    
    print(f"\n{'='*80}")
    print(f"BINARY CLASSIFICATION (Threshold: {rating_threshold})")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted Not Advised  Predicted Advised")
    print(f"Actual Not Advised        {conf_matrix[0,0]:5d}                {conf_matrix[0,1]:5d}")
    print(f"Actual Advised            {conf_matrix[1,0]:5d}                {conf_matrix[1,1]:5d}")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
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
    
    # Plot 4: Error vs Actual Rating
    axes[1, 1].scatter(all_actual, all_errors, alpha=0.6, s=50, color='mediumpurple')
    axes[1, 1].axhline(np.mean(all_errors), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(all_errors):.3f}')
    axes[1, 1].set_xlabel('Actual Rating', fontsize=11)
    axes[1, 1].set_ylabel('Absolute Error', fontsize=11)
    axes[1, 1].set_title('Error vs Actual Rating', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 5.5)
    
    categories = ['True Not Advised\n(TN)', 'False Advised\n(FP)', 
                  'False Not Advised\n(FN)', 'True Advised\n(TP)']
    values = [conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]]
    colors = ['lightgreen', 'salmon', 'salmon', 'lightgreen']
    
    bars = axes[2, 0].bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
    axes[2, 0].set_ylabel('Count', fontsize=11)
    axes[2, 0].set_title(f'Binary Classification Results (Threshold: {rating_threshold})', 
                         fontsize=12, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[2, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(value)}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # NEW: Plot 6: Binary Classification Metrics
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [accuracy, precision, recall, f1]
    
    bars = axes[2, 1].barh(metrics_names, metrics_values, color='steelblue', edgecolor='black', alpha=0.7)
    axes[2, 1].set_xlabel('Score', fontsize=11)
    axes[2, 1].set_title('Binary Classification Metrics', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlim(0, 1)
    axes[2, 1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        width = bar.get_width()
        axes[2, 1].text(width, bar.get_y() + bar.get_height()/2.,
                       f'{value:.3f}',
                       ha='left', va='center', fontsize=10, fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved as: {plot_filename}")
    print(f"{'='*80}\n")
    
    return all_errors, all_actual, all_predicted, all_actual_binary, all_predicted_binary

def plot_metrics(metrics, params: dict):


    model_path = params.get('MODEL_PATH', './default_model.pt')
    model_filename = os.path.basename(model_path)
    model_name_base = os.path.splitext(model_filename)[0]
    plot_filename = f"{model_name_base}_loss.png" 
    
    epochs = [m['epoch'] for m in metrics]
    losses = [m['loss'] for m in metrics]
    train_rmses = [m['train_rmse'] for m in metrics]
    val_rmses = [m['val_rmse'] for m in metrics]
    val_rmses_mean = [m['current_window_mean'] for m in metrics]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Subplot 1: Training Loss
    axes[0].plot(epochs, losses, label='Train Loss (MSE)', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss per Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: RMSE (Train vs. Validation)
    
    axes[1].plot(epochs, train_rmses, label='Train RMSE', color='orange')
    axes[1].plot(epochs, val_rmses, label='Validation RMSE', color='red')
    axes[1].plot(epochs, val_rmses_mean, label='Mean Validation RMSE', color='blue')

    
    # Find the best validation MEAN RMSE for annotation
    if val_rmses:
        best_val = min(val_rmses_mean)
        best_epoch = epochs[val_rmses_mean.index(best_val)]
        axes[1].axvline(x=best_epoch, color='gray', linestyle='--', label=f'Best Val Epoch ({best_epoch})')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE per Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"\n✓ Metrics plot saved as: {plot_filename}")



def log_parameters(params: dict):

    # 1. Get current timestamp
    #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 2. Extract model name from MODEL_PATH
    model_path = params.get('MODEL_PATH', './default_model.pt')
    
    # Get the filename (e.g., 'best_model.pt')
    model_filename = os.path.basename(model_path)
    
    # Remove the extension (e.g., 'best_model')
    model_name_base = os.path.splitext(model_filename)[0]
    
    # 3. Construct the log file name
    log_filename = f"{model_name_base}_parameters.txt"
    
    log_data = f"{model_name_base}_params_data.json"

    # 4. Write parameters to the log file
    try:
        with open(log_filename, 'w') as f:
            f.write(f"--- Training Configuration Log ---\n")
            #f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model Path: {model_path}\n")
            f.write("-" * 40 + "\n")
            
            # Sort parameters alphabetically for a consistent log format
            for key in sorted(params.keys()):
                value = params[key]
                # Use json.dumps for clean printing of lists/dicts (like NUM_NEIGHBORS)
                f.write(f"{key:<30}: {json.dumps(value)}\n") 
        
        print(f"\n✓ Configuration saved to log file: {log_filename}")
    except IOError as e:
        print(f"Error saving log file {log_filename}: {e}")
    
    with open(log_data, 'w') as f:
        json.dump(params, f)
