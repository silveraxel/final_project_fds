#Implicit model means that we dont use the actual rating but some labels that tell us if a user have seen a film
#and gave to it a good rating(in our case >=4)
#Before runnig this code check the path to the dataset
#to run this code just run this line python final_implicit.py
#To apply gridsearch just run this line  python final_implicit.py --grid_search 1
import argparse
import os
import random
import math
import time
import json
from itertools import product
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.nn import MultiheadAttention
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

# -----------------------------
# Utilities function
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_year(title):#example of a title: "Toy Story (1995)"
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else np.nan

def remove_year_from_title(title):
    return re.sub(r"\s*\(\d{4}\)", "", title)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# Compute MiniLM title embeddings (cached recommended)
# -----------------------------
def compute_title_embeddings(movie_df, model_name='all-MiniLM-L6-v2', device='cpu', cache_path=None):
    # We encode movie titles using a pretrained MiniLM model to capture
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Loading cached MiniLM embeddings from {cache_path}")
        return np.load(cache_path)
    print("Instantiating SentenceTransformer:", model_name)
    model = SentenceTransformer(model_name, device=device)
    titles = movie_df['title'].astype(str).tolist()
    emb = model.encode(titles, convert_to_tensor=True, show_progress_bar=True)
    emb_np = emb.cpu().numpy().astype(np.float32)
    if cache_path is not None:
        np.save(cache_path, emb_np)
    return emb_np

# -----------------------------
# Data loading & preprocessing
# -----------------------------
def load_rating(ratings_path: str = "ratings.csv", strategy: str = "last"):
    df = pd.read_csv(ratings_path, sep=",")
    assert {"userId", "movieId", "rating"}.issubset(df.columns), "ratings.csv missing required columns"
    # To remove duplicate ratings of a movie by a single user
    if df.duplicated(subset=["userId", "movieId"]).any():
        if strategy == "last":
            df = df.sort_values("timestamp").drop_duplicates(subset=["userId", "movieId"], keep="last")
        elif strategy == "avg":
            df = df.groupby(["userId", "movieId"], as_index=False)["rating"].mean()
        else:
            raise ValueError("strategy must be 'last' or 'avg'")
    return df

def load_movie(movies_path="movies.csv"):
    movies = pd.read_csv(movies_path, sep=",")
    movies["genre_list"] = movies["genres"].apply(lambda g: g.split("|") if isinstance(g, str) else [])
    movies["year"] = movies["title"].apply(extract_year)
    # Create cleaned title
    movies["title"] = movies["title"].apply(remove_year_from_title)
    # Hot-encoding of genres
    mlb = MultiLabelBinarizer()
    genres_multi_hot = mlb.fit_transform(movies['genre_list'])
    genres_multi_hot = pd.DataFrame(genres_multi_hot, columns=mlb.classes_, index=movies.index)
    movies = pd.concat([movies, genres_multi_hot], axis=1)
    genre_cols = list(mlb.classes_)
    return movies[['movieId', 'year', 'title'] + genre_cols], genre_cols

# Function to split dataset in train/test/val
def leave_k_out_train_val_test(df: pd.DataFrame, k_test: float = 0.1, k_val: float = 0.1):
    # Splitting the dataset s.t. past interaction are used to train the model and the latest as test and validation 
    df = df.sort_values(["user_idx", "timestamp"]) if "timestamp" in df.columns else df
    def get_split_counts(n, k_val, k_test):
        if isinstance(k_val, float) and 0 < k_val < 1:
            k_val = int(np.ceil(k_val * n))
        if isinstance(k_test, float) and 0 < k_test < 1:
            k_test = int(np.ceil(k_test * n))
        return k_val, k_test
    val_indices = []
    test_indices = []
    for user, group in df.groupby("user_idx"):
        k_v, k_t = get_split_counts(len(group), k_val, k_test)
        k_v = min(k_v, len(group))
        k_t = min(k_t, len(group) - k_v)
        val_indices.extend(group.tail(k_v).index)
        test_indices.extend(group.iloc[-(k_v + k_t):-k_v].index) if k_t > 0 else None
    val_df = df.loc[val_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    train_df = df.drop(val_indices + test_indices).reset_index(drop=True)
    return train_df, test_df, val_df

def map_and_binarize(raw_rating: pd.DataFrame, raw_movie: pd.DataFrame, genre_cols, k_test: float = 0.1, k_val: float = 0.1, threshold: float = 4.0):
    # Feature extraction for the implicit reccomendation system
    df = raw_rating.copy()
    df["label"] = (df["rating"] >= threshold).astype(int)
    users = np.sort(df["userId"].unique())
    items = np.sort(df["movieId"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {i: j for j, i in enumerate(items)}
    df["user_idx"] = df["userId"].map(user2idx)
    df["item_idx"] = df["movieId"].map(item2idx)
    df = df.merge(raw_movie[['movieId', 'year'] + genre_cols], on='movieId', how='left')
    if df['year'].isnull().any():
        df['year'] = df['year'].fillna(df['year'].median())
    # Split dataset in train/test/val
    train_df, test_df, val_df = leave_k_out_train_val_test(df, k_test=k_test, k_val=k_val)
    # Dense features
    movie_stats = (train_df.groupby("item_idx")["rating"].agg(avg_movie_rating="mean", count_movie_rating="count").reset_index())
    user_stats = (train_df.groupby("user_idx")["rating"].agg(avg_user_rating="mean", count_user_rating="count").reset_index())
    train_df = train_df.merge(movie_stats, on="item_idx", how="left").merge(user_stats, on="user_idx", how="left")
    val_df = val_df.merge(movie_stats, on="item_idx", how="left").merge(user_stats, on="user_idx", how="left")
    test_df = test_df.merge(movie_stats, on="item_idx", how="left").merge(user_stats, on="user_idx", how="left")
    # Filling null value to prevent the crashing of the model and also too help cold start
    global_movie_avg = train_df["avg_movie_rating"].mean() if not train_df["avg_movie_rating"].isnull().all() else 0.0
    global_movie_count = train_df["count_movie_rating"].mean() if "count_movie_rating" in train_df.columns else 0.0
    global_user_avg = train_df["avg_user_rating"].mean() if not train_df["avg_user_rating"].isnull().all() else 0.0
    global_user_count = train_df["count_user_rating"].mean() if "count_user_rating" in train_df.columns else 0.0
    for df_ in (train_df, val_df, test_df):
        df_["avg_movie_rating"] = df_["avg_movie_rating"].fillna(global_movie_avg)
        df_["count_movie_rating"] = df_["count_movie_rating"].fillna(global_movie_count)
        df_["avg_user_rating"] = df_["avg_user_rating"].fillna(global_user_avg)
        df_["count_user_rating"] = df_["count_user_rating"].fillna(global_user_count)
    agg_cols = ["year", "avg_movie_rating", "count_movie_rating", "avg_user_rating", "count_user_rating"]
    agg_scaler = StandardScaler()
    train_df[agg_cols] = agg_scaler.fit_transform(train_df[agg_cols])
    val_df[agg_cols] = agg_scaler.transform(val_df[agg_cols])
    test_df[agg_cols] = agg_scaler.transform(test_df[agg_cols])
    num_users = len(users)
    num_items = len(items)
    return train_df, val_df, test_df, num_users, num_items, agg_cols, user2idx, item2idx

# -----------------------------
# Negative sampling utilities
# -----------------------------
# Dynamically generates negative samples at each epoch by pairing users with items they have not interacted with
# Prevents the model from overfitting to a fixed set of negatives
class DynamicNegativeSampler:
    def __init__(self, train_pos_df: pd.DataFrame, num_items: int, num_neg: int = 4, feature_cols=None, seed: int = 42):
        self.train_pos_df = train_pos_df.copy()
        self.feature_cols = feature_cols or []
        self.num_items = num_items
        self.num_neg = num_neg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.pos_df = self.train_pos_df[self.train_pos_df.label == 1]
        self.neg_true_df = self.train_pos_df[self.train_pos_df.label == 0]
        self.user_pos = self.pos_df.groupby("user_idx").item_idx.apply(set).to_dict()

    def resample(self):
        neg_rows = []
        all_items = np.arange(self.num_items)
        item_feats = (self.train_pos_df[["item_idx"] + self.feature_cols + ["year", "avg_movie_rating", "count_movie_rating"]]
                      .drop_duplicates("item_idx").set_index("item_idx"))
        user_feats = (self.train_pos_df[["user_idx", "avg_user_rating", "count_user_rating"]]
                      .drop_duplicates("user_idx").set_index("user_idx"))
        # For each positive interaction, sample K unseen items for the same user.
        for (_, row) in self.pos_df.iterrows():
            u = int(row.user_idx)
            watched = self.user_pos.get(u, set())
            available = np.setdiff1d(all_items, np.array(list(watched)))
            if len(available) == 0:
                continue
            k = min(self.num_neg, len(available))
            sampled = self.rng.choice(available, k, replace=False)
            for it in sampled:
                neg_rows.append({"user_idx": u, "item_idx": it, "label": 0})
        neg_df = pd.DataFrame(neg_rows)
        if not neg_df.empty and item_feats is not None:
            neg_df = neg_df.merge(item_feats.reset_index(), on="item_idx", how="left")
        if not neg_df.empty and user_feats is not None:
            neg_df = neg_df.merge(user_feats.reset_index(), on="user_idx", how="left")
        # Filling null value to prevent the crashing of the model and also too help cold start
        global_movie_avg = self.train_pos_df["avg_movie_rating"].mean() if not self.train_pos_df["avg_movie_rating"].isnull().all() else 0.0
        global_movie_count = self.train_pos_df["count_movie_rating"].mean() if "count_movie_rating" in self.train_pos_df.columns else 0.0
        global_user_avg = self.train_pos_df["avg_user_rating"].mean() if not self.train_pos_df["avg_user_rating"].isnull().all() else 0.0
        global_user_count = self.train_pos_df["count_user_rating"].mean() if "count_user_rating" in self.train_pos_df.columns else 0.0
        neg_df["avg_movie_rating"] = neg_df["avg_movie_rating"].fillna(global_movie_avg)
        neg_df["count_movie_rating"] = neg_df["count_movie_rating"].fillna(global_movie_count)
        neg_df["avg_user_rating"] = neg_df["avg_user_rating"].fillna(global_user_avg)
        neg_df["count_user_rating"] = neg_df["count_user_rating"].fillna(global_user_count)
        combined = pd.concat([self.pos_df, neg_df], ignore_index=True, sort=False)
        combined = combined.sample(frac=1.0, random_state=int(self.rng.integers(0, 2 ** 31 - 1))).reset_index(drop=True)
        return combined

# -----------------------------
# Dataset
# -----------------------------
class RecDataset(Dataset):
    # Torch dataset that aligns all user, item, and side-information features into a single training sample
    def __init__(self, df: pd.DataFrame, agg_cols, genre_cols, minilm_matrix=None):
        self.users = df['user_idx'].values.astype(np.int64)
        self.items = df['item_idx'].values.astype(np.int64)
        self.labels = df['label'].values.astype(np.float32)
        self.agg_cols = agg_cols or []
        self.genre_cols = genre_cols or []
        if len(self.genre_cols) > 0:
            self.genre_cols = df[self.genre_cols].fillna(0).values.astype(np.float32)
        else:
            self.genre_cols = None
        if len(self.agg_cols) > 0:
            self.agg_cols = df[self.agg_cols].fillna(0).values.astype(np.float32)
        else:
            self.agg_cols = None
        # minilm_matrix shape (num_items, title_dim)
        if minilm_matrix is not None:
            item_idx_order = df['item_idx'].values
            self.minilm = minilm_matrix[item_idx_order]
        else:
            self.minilm = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'item': torch.tensor(self.items[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        if self.genre_cols is not None:
            sample['genre_cols'] = torch.tensor(self.genre_cols[idx], dtype=torch.float)
        else:
            sample['genre_cols'] = torch.tensor([], dtype=torch.float)
        if self.agg_cols is not None:
            sample['agg_cols'] = torch.tensor(self.agg_cols[idx], dtype=torch.float)
        else:
            sample['agg_cols'] = torch.tensor([], dtype=torch.float)
        if self.minilm is not None:
            sample['minilm'] = torch.tensor(self.minilm[idx], dtype=torch.float)
        else:
            sample['minilm'] = torch.tensor([], dtype=torch.float)
        return sample

# -----------------------------
# DeepFM
# -----------------------------
class DeepFM(nn.Module):
    def __init__(self, num_genres, num_agg, num_users, num_items, title_dim=384, emb_dim=32, mlp_dims=(64, 32),
                 dropout=0.2, use_batchnorm=True, emb_dropout=0.05, init_method='kaiming'):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity()
        self.genre_emb = nn.EmbeddingBag(num_genres, emb_dim, mode="mean")
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        # project MiniLM (title) vector to emb_dim
        # proj from 
        self.minilm_proj =  nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)  
        ) 
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        input_dim = emb_dim * 4 + num_agg
        layers = []
        last = input_dim
        for h in mlp_dims:
            layers.append(nn.Linear(last, h))
            if use_batchnorm: layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0.0: layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.dnn = nn.Sequential(*layers)
        self.attn = MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True)
        self.init_method = init_method
        self._init_weights()

    def _init_weights(self):
        if self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.user_emb.weight)
            nn.init.xavier_uniform_(self.item_emb.weight)
            nn.init.xavier_uniform_(self.genre_emb.weight)
        else:
            nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.genre_emb.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.user_bias.weight, 0.0)
        nn.init.constant_(self.item_bias.weight, 0.0)
        for m in self.dnn.modules():
            if isinstance(m, nn.Linear):
                if self.init_method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        for layer in self.minilm_proj:
            if isinstance(layer, nn.Linear):
                if self.init_method == 'xavier':
                    nn.init.xavier_uniform_(layer.weight)
                else:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0) 

    def fm_interaction(self, vectors): # FM layer: captures low-order feature interactions
        sum_fields = torch.sum(vectors, dim=1)
        sum_square = sum_fields * sum_fields
        square_sum = torch.sum(vectors * vectors, dim=1)
        interactions = 0.5 * (sum_square - square_sum)
        return torch.sum(interactions, dim=1, keepdim=True)

    def forward(self, user, item, genre_multi_hot, dense_feats, minilm_vec):
        u_e = self.user_emb(user)
        i_e = self.item_emb(item)
        minilm_vec = F.normalize(minilm_vec, dim=1)
        ml = self.minilm_proj(minilm_vec)
        
        if self.emb_dropout:
            u_e = self.emb_dropout(u_e)
            i_e = self.emb_dropout(i_e)
        genre_embeddings = self.genre_emb.weight.unsqueeze(0).expand(user.size(0), -1, -1)
        query = (u_e + i_e).unsqueeze(1)
        # Attention over genres: learns which genres matter for a user-item pair
        attn_out, _ = self.attn(query, genre_embeddings, genre_embeddings)
        g_emb = attn_out.squeeze(1)
        u_b = self.user_bias(user)
        i_b = self.item_bias(item)
        fm_vecs = torch.stack([u_e, i_e, g_emb,ml], dim=1)
        fm_out = self.fm_interaction(fm_vecs)
        concat = torch.cat([u_e, i_e, g_emb, dense_feats,ml], dim=1)
        # DNN: captures higher-order nonlinear interactions
        dnn_out = self.dnn(concat)
        out = u_b + i_b + fm_out + dnn_out
        return out.squeeze(1)

# -----------------------------
# Evaluation metrics
# -----------------------------
# HR@K and NDCG@K are ranking-based metrics suitable for implicit feedback.
# They measure whether the true held-out item is ranked highly among a large set of unobserved candidate items
def hit_rate_at_k(recs, truth, k):
    return int(truth in recs[:k])

def ndcg_at_k(recs, truth, k):
    if truth in recs[:k]:
        rank = recs.index(truth)
        return 1.0 / math.log2(rank + 2)
    return 0.0

@torch.no_grad()
def evaluate_model(model, train_user_pos: dict, test_df: pd.DataFrame, num_items: int, genre_matrix, agg_matrix, minilm_matrix, k: int = 10, device: str = 'cpu'):
    model.to(device)
    model.eval()
    users = test_df['user_idx'].unique()
    HRs = []
    NDCGs = []
    genre_tensor = torch.tensor(genre_matrix, dtype=torch.float, device=device)
    agg_tensor = torch.tensor(agg_matrix, dtype=torch.float, device=device)
    minilm_tensor = torch.tensor(minilm_matrix, dtype=torch.float, device=device)
    for u in users:
        truth = int(test_df[test_df['user_idx'] == u]['item_idx'].iloc[0])
        seen = train_user_pos.get(u, set())
        candidates = np.setdiff1d(np.arange(num_items, dtype=np.int64), np.array(list(seen), dtype=np.int64))
        if len(candidates) == 0:
            HRs.append(0); NDCGs.append(0); continue
        user_tensor = torch.tensor([u] * len(candidates), dtype=torch.long, device=device)
        item_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        g = genre_tensor[item_tensor]
        a = agg_tensor[item_tensor]
        ml = minilm_tensor[item_tensor]
        scores = model(user_tensor, item_tensor, g, a, ml).cpu().numpy()
        ranked_idx = np.argsort(-scores)
        recs = [candidates[i] for i in ranked_idx.tolist()]
        HRs.append(hit_rate_at_k(recs, truth, k))
        NDCGs.append(ndcg_at_k(recs, truth, k))
    return float(np.mean(HRs)), float(np.mean(NDCGs))

# -----------------------------
# Training loop with dynamic negatives + early stopping + checkpoint
# -----------------------------
def train_with_dynamic_negatives(agg_cols, genre_cols, genre_matrix, agg_matrix, minilm_matrix, model: nn.Module,
                                 train_pos_df: pd.DataFrame, val_df: pd.DataFrame, num_users: int, num_items: int,
                                 epochs: int = 20, batch_size: int = 1024, lr: float = 1e-3, weight_decay: float = 1e-5,
                                 num_neg: int = 4, device: str = 'cpu', patience: int = 5, checkpoint_dir: str = 'checkpoints'):
    ensure_dir(checkpoint_dir)
    sampler = DynamicNegativeSampler(train_pos_df, num_items, num_neg=num_neg, feature_cols=genre_cols)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    loss_fn = nn.BCEWithLogitsLoss()
    best_ndcg = -1.0; best_epoch = 0; history = []
    train_user_pos_map = train_pos_df[train_pos_df['label'] == 1].groupby('user_idx')['item_idx'].apply(set).to_dict()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_aug = sampler.resample()
        dataset = RecDataset(train_aug, agg_cols, genre_cols, minilm_matrix=minilm_matrix)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        model.to(device); model.train()
        running_loss = 0.0; n_batches = 0
        for batch in loader:
            users = batch['user'].to(device)
            items = batch['item'].to(device)
            labels = batch['label'].to(device)
            g = batch['genre_cols'].to(device)
            a = batch['agg_cols'].to(device)
            ml = batch['minilm'].to(device)
            preds = model(users, items, g, a, ml)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running_loss += loss.item(); n_batches += 1
        avg_loss = running_loss / max(1, n_batches)
        hr, ndcg = evaluate_model(model, train_user_pos_map, val_df, num_items, genre_matrix, agg_matrix, minilm_matrix, k=10, device=device)
        scheduler.step(ndcg)
        history.append({'epoch': ep, 'loss': avg_loss, 'hr_val': hr, 'ndcg_val': ndcg, 'time': time.time() - t0})
        print(f"Epoch {ep:02d} | loss={avg_loss:.4f} | HR@10_val={hr:.4f} | NDCG@10_val={ndcg:.4f} | time={history[-1]['time']:.1f}s")
        if ndcg > best_ndcg:
            best_ndcg = ndcg; best_epoch = ep
            ckpt_path = os.path.join(checkpoint_dir, f"best_model_epoch{ep}.pt")
            torch.save({'epoch': ep, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt_path)
        if ep - best_epoch >= patience:
            print(f"Early stopping triggered (no improvement in {patience} epochs). Best epoch: {best_epoch}, best NDCG: {best_ndcg:.4f}")
            break
    return model, pd.DataFrame(history)

# -----------------------------
# Grid Search
# -----------------------------
def run_grid_search(agg_cols, genre_cols, genre_matrix, agg_matrix, minilm_matrix, df, num_users, num_items, test_df,val_df, param_grid, out_csv='grid_search_results.csv', device='cpu'):
    rows = []; total = 0
    if 'mlp_dims' in param_grid: param_grid['mlp_dims'] = [tuple(v) for v in param_grid['mlp_dims']]
    for vals in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), vals)); total += 1
        print(f"\n[Grid] Starting experiment {total} with params: {params}")
        seed_everything(params.get('seed', 42))
        model = DeepFM(len(genre_cols), len(agg_cols), num_users, num_items, title_dim=params.get('title_dim', 384), emb_dim=params['embed_dim'], mlp_dims=tuple(params['mlp_dims']))
        model, history = train_with_dynamic_negatives(agg_cols, genre_cols, genre_matrix, agg_matrix, minilm_matrix, model=model, train_pos_df=df, val_df=val_df, num_users=num_users, num_items=num_items, epochs=params['epochs'], batch_size=params['batch_size'], lr=params['lr'], weight_decay=params['weight_decay'], num_neg=params['num_neg'], device=device, patience=params.get('patience', 5))
        train_user_pos_map = df[df['label'] == 1].groupby('user_idx')['item_idx'].apply(set).to_dict()
        hr, ndcg = evaluate_model(model, train_user_pos_map, test_df, num_items, genre_matrix, agg_matrix, minilm_matrix, k=10, device=device)
        result = {**params, 'hr': hr, 'ndcg': ndcg}; rows.append(result)
        df_rows = pd.DataFrame(rows); df_rows.to_csv(out_csv, index=False)
        print(f"[Grid] Result: HR@10={hr:.4f}, NDCG@10={ndcg:.4f} (saved to {out_csv})")
    
    # --- AFTER finishing all experiments ---
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_csv, index=False)

    # Find and save best params by NDCG
    if len(results_df) > 0:
        best_idx = results_df['ndcg'].idxmax()
        best_row = results_df.loc[best_idx]
        best_params = best_row.to_dict()

        with open('best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)

        print("\n[Grid] Best parameters found:")
        print(json.dumps(best_params, indent=2))

    return results_df

# -----------------------------
# Main CLI
# -----------------------------
def main(args):
    seed_everything(args.seed)
    raw_rating = load_rating(args.ratings)
    raw_movie, genre_cols = load_movie(args.movies)
    # Compute MiniLM embeddings
    print("Computing MiniLM embeddings for titles (or loading cache)...") 
    device_name = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    # Optional: pass cache path via args if you want to reuse
    cache_path = args.minilm_cache if hasattr(args, 'minilm_cache') else None
    minilm_emb = compute_title_embeddings(raw_movie, device=device_name, cache_path=cache_path)  # numpy (n_movies, 384)
    train_pos_df, val_df, test_df, num_users, num_items, agg_cols, u2i, m2i = map_and_binarize(raw_rating, raw_movie, genre_cols, k_test=0.1, k_val=0.1, threshold=args.threshold)
    print(f"Users: {num_users}, Items: {num_items}, Train interactions: {len(train_pos_df)}, Val interactions: {len(val_df)}, Test interactions: {len(test_df)}")
    feature_cols = genre_cols + agg_cols
    item_feat_df = train_pos_df[["item_idx"] + feature_cols].drop_duplicates("item_idx").set_index('item_idx')
    missing = set(range(num_items)) - set(item_feat_df.index.tolist())
    for m in missing:
        item_feat_df.loc[m] = [0.0] * (len(feature_cols))
    item_feat_df = item_feat_df.sort_index()
    genre_matrix = item_feat_df[genre_cols].values.astype(np.float32)
    agg_matrix = item_feat_df[agg_cols].values.astype(np.float32)
    # align minilm_emb (raw_movie order) to item_idx order using m2i mapping
    raw_movie_idx = {mid: i for i, mid in enumerate(raw_movie['movieId'].values)}
    minilm_matrix = np.zeros((num_items, minilm_emb.shape[1]), dtype=np.float32)
    for mid, item_idx in m2i.items():
        if mid in raw_movie_idx:
            minilm_matrix[item_idx] = minilm_emb[raw_movie_idx[mid]]

    if args.grid_search:#if runned with option: --grid_search 1
        param_grid = {
            'embed_dim': args.grid_embed_dim,
            'mlp_dims': args.grid_mlp_dims,
            'epochs': args.grid_epochs,
            'batch_size': args.grid_batch_size,
            'lr': args.grid_lr,
            'weight_decay': args.grid_weight_decay,
            'num_neg': args.grid_num_neg,
            'patience': args.grid_patience,
            'seed': args.grid_seed
        }
        # Convert single values into lists if necessary
        for k, v in list(param_grid.items()):
            if not isinstance(v, (list, tuple)):
                param_grid[k] = [v]
        # Ensure mlp dims are tuples
        if 'mlp_dims' in param_grid:
            param_grid['mlp_dims'] = [tuple(m) for m in param_grid['mlp_dims']]

        print("Starting grid search with grid:")
        print(json.dumps(param_grid, indent=2))
        results =run_grid_search(agg_cols, genre_cols, genre_matrix, agg_matrix, minilm_matrix, train_pos_df, num_users, num_items, test_df,val_df, param_grid, out_csv='grid_search_results.csv', device='cpu')        
        print("Grid search finished. Results:\n", results)
        return
    
    # Quick execution (model + train + evaluate)
    model = DeepFM(len(genre_cols), len(agg_cols), num_users, num_items, title_dim=minilm_matrix.shape[1], emb_dim=args.embed_dim, mlp_dims=tuple(args.mlp_dims), dropout=0.2, use_batchnorm=True, emb_dropout=0.05, init_method='kaiming')
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    model.to(device)
    model, history = train_with_dynamic_negatives(agg_cols, genre_cols, genre_matrix, agg_matrix, minilm_matrix, model, train_pos_df, val_df, num_users, num_items, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, num_neg=args.num_neg, device=device, patience=args.patience, checkpoint_dir=args.checkpoint_dir)
    train_user_pos_map = train_pos_df[train_pos_df['label'] == 1].groupby('user_idx')['item_idx'].apply(set).to_dict()
    hr, ndcg = evaluate_model(model, train_user_pos_map, test_df, num_items, genre_matrix, agg_matrix, minilm_matrix, k=args.K, device=device)
    print(f"Final TEST HR@{args.K}: {hr:.4f} | NDCG@{args.K}: {ndcg:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings', type=str, default='../dataset/ratings.csv')
    parser.add_argument('--movies', type=str, default='../dataset/movies.csv')
    parser.add_argument('--threshold', type=float, default=4.0, help='binarization threshold (>=)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--mlp_dims', type=int, nargs='+', default=[128,64,32])
    parser.add_argument('--num_neg', type=int, default=4)#8
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--grid_search', type=int, default=0)
    parser.add_argument('--grid_embed_dim', type=int, nargs='*', default=[24,32,48,96])
    parser.add_argument('--grid_mlp_dims', type=int, nargs='*', default=[[64,32],[128, 64],[256, 128],[256, 128, 64],[128,64,32]])#[(128, 64),(256, 128),(256, 128, 64)]
    parser.add_argument('--grid_epochs', type=int, nargs='*', default=[30])
    parser.add_argument('--grid_batch_size', type=int, nargs='*', default=[1024, 2048])
    parser.add_argument('--grid_lr', type=float, nargs='*', default=[7e-4,5e-4,3e-4])
    parser.add_argument('--grid_weight_decay', type=float, nargs='*', default=[1e-6, 1e-5, 5e-5])
    parser.add_argument('--grid_num_neg', type=int, nargs='*', default=[4])
    parser.add_argument('--grid_seed', type=int, nargs='*', default=[42])
    parser.add_argument('--grid_patience', type=int, default=5)
    parser.add_argument('--grid_out', type=str, default='grid_search_results.csv')
    # caching path for minilm emb (optional)
    parser.add_argument('--minilm_cache', type=str, default='minilm_title_emb.npy')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'quick'])
    args = parser.parse_args()
    if args.mode == 'quick':
        args.epochs = 5
        args.batch_size = 2048
    main(args)