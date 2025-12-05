#SISTEMA DI RACCOMANDAZIONE IMPLICITO QUINDI NON PREDICE UN RATING PER UN FILM MA AL POSTO DEI RATING USA DEI 
#LABEL 1(film visto e apprezzato con una valutazione maggiore o uguale a 4)/0(film non visto o con valutazione sotto al 4)

#NOTE: QUESTO MODELLO PER ORA USA DELLE FEATURE BASILARI E CI POTREBBERO ESSERE DEI MIGLIORAMENTI, TO DO:
# BATCH EVALUATION
# ADAPTIVE LR, ONE CYCLE LR
# MULTIHEAD ATTENTION
# MULTITASKING, UNIRE UN MODELLO IMPLICITO E UNO ESPLICITO
# AGGIUNGERE DELLE FEATURES (EMBEDDING DEL TITOLO, AVG E COUNT DEI RATINGS PER I MOVIE E GLI USER)

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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else np.nan


def remove_year_from_title(title):
    return re.sub(r"\s*\(\d{4}\)", "", title)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Data loading & preprocessing
# -----------------------------

def load_movielens(ratings_path: str = "ratings.csv", strategy: str = "last"):
    df = pd.read_csv(ratings_path, sep=",")
    # minimal safety checks
    assert {"userId", "movieId", "rating"}.issubset(df.columns), "ratings.csv missing required columns"
    if df.duplicated(subset=["userId", "movieId"]).any():
        if strategy == "last":
            # sort by timestamp
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
    movies["title"] = movies["title"].apply(remove_year_from_title)

    # MultiLabelBinarizer for genres
    mlb = MultiLabelBinarizer()
    genres_multi_hot = mlb.fit_transform(movies['genre_list'])
    genres_multi_hot = pd.DataFrame(genres_multi_hot, columns=mlb.classes_, index=movies.index)
    movies = pd.concat([movies, genres_multi_hot], axis=1)
    genre_cols = list(mlb.classes_)
    # return movie features (movieId -> features)
    return movies[['movieId', 'year'] + genre_cols], genre_cols


def leave_one_out_split(df: pd.DataFrame):
    # per user, hold out the latest interaction (by timestamp) as test
    df = df.sort_values(["user_idx", "timestamp"]) if "timestamp" in df.columns else df
    test_idx = df.groupby('user_idx').tail(1).index
    test_df = df.loc[test_idx].reset_index(drop=True)
    train_df = df.drop(test_idx).reset_index(drop=True)
    return train_df, test_df


def map_and_binarize(raw_rating: pd.DataFrame, raw_movie: pd.DataFrame, genre_cols, threshold: float = 4.0):
    df = raw_rating.copy()
    df["label"] = (df["rating"] >= threshold).astype(int)

    users = np.sort(df["userId"].unique())
    items = np.sort(df["movieId"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {i: j for j, i in enumerate(items)}

    df["user_idx"] = df["userId"].map(user2idx)
    df["item_idx"] = df["movieId"].map(item2idx)

    # movie features
    df = df.merge(
        raw_movie[['movieId', 'year'] + genre_cols],
        on='movieId',
        how='left'
    )

    # fill missing year with median (safer than dropping)
    if df['year'].isnull().any():
        df['year'] = df['year'].fillna(df['year'].median())

    scaler = StandardScaler()
    train_df, test_df = leave_one_out_split(df)

    # fit scaler only on train years
    train_df['year'] = scaler.fit_transform(train_df[['year']])
    test_df['year'] = scaler.transform(test_df[['year']])

    num_users = len(users)
    num_items = len(items)
    return train_df, test_df, num_users, num_items, user2idx, item2idx


# -----------------------------
# Negative sampling utilities
# -----------------------------
class DynamicNegativeSampler:
    """Generate negative samples dynamically given train positives.
    Use .resample() each epoch to produce a new augmented training df.
    Returns a DataFrame with positive and sampled negative rows.
    """

    def __init__(self, train_pos_df: pd.DataFrame, num_items: int, num_neg: int = 4, feature_cols=None, seed: int = 42):
        self.train_pos_df = train_pos_df.copy()
        self.feature_cols = feature_cols or []
        self.num_items = num_items
        self.num_neg = num_neg
        self.seed = seed
        self.user_pos = self.train_pos_df[self.train_pos_df['label'] == 1].groupby('user_idx')['item_idx'].apply(set).to_dict()
        self.users = list(self.user_pos.keys())
        self.rng = random.Random(seed)

    def resample(self):
        neg_rows = []
        for u in self.users:
            watched = self.user_pos.get(u, set())
            # produce candidate pool
            available = [i for i in range(self.num_items) if i not in watched]
            if len(available) == 0:
                continue
            k = min(self.num_neg, len(available))
            sampled = self.rng.sample(available, k)
            for it in sampled:
                neg_rows.append({"user_idx": u, "item_idx": it, "label": 0})

        neg_df = pd.DataFrame(neg_rows)

        # attach item features to negatives (if available) and ensure positives keep their features
        if not neg_df.empty and self.feature_cols:
            neg_df = neg_df.merge(
                self.train_pos_df[["item_idx"] + self.feature_cols].drop_duplicates("item_idx"),
                on="item_idx",
                how="left"
            )

        pos_df = self.train_pos_df[self.train_pos_df['label'] == 1][['user_idx', 'item_idx', 'label'] + self.feature_cols]
        combined = pd.concat([pos_df, neg_df], ignore_index=True, sort=False)

        # fill any missing feature columns (rare) with zeros
        for c in self.feature_cols:
            if c not in combined.columns:
                combined[c] = 0.0

        # shuffle
        combined = combined.sample(frac=1.0, random_state=self.rng.randint(0, 2 ** 31 - 1)).reset_index(drop=True)
        return combined


# -----------------------------
# Dataset
# -----------------------------
class RecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols):
        self.users = df['user_idx'].values.astype(np.int64)
        self.items = df['item_idx'].values.astype(np.int64)
        self.labels = df['label'].values.astype(np.float32)
        self.feature_cols = feature_cols or []
        if len(self.feature_cols) > 0:
            self.features = df[self.feature_cols].fillna(0).values.astype(np.float32)
        else:
            self.features = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'item': torch.tensor(self.items[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        if self.features is not None:
            sample['feat'] = torch.tensor(self.features[idx], dtype=torch.float)
        else:
            sample['feat'] = torch.tensor([], dtype=torch.float)
        return sample


# -----------------------------
# DeepFM (correct FM interaction term)
# -----------------------------
class DeepFM(nn.Module):
    def __init__(self, num_features, num_users, num_items, emb_dim=32, mlp_dims=(64, 32),
                 dropout=0.2, use_batchnorm=True, emb_dropout=0.05,
                 init_method='kaiming'):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_dropout=emb_dropout
        self.num_features = num_features
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        # first-order (linear) terms: user/item biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # DNN (on concatenated embeddings + features)
        input_dim = emb_dim * 2 + num_features
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

        # store init choice
        self.init_method = init_method
        self._init_weights()

        # initialize
    def _init_weights(self):
        # Embeddings: small normal (or xavier)
        if self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.user_emb.weight)
            nn.init.xavier_uniform_(self.item_emb.weight)
        else:
            # recommended for ReLU DNN
            nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)

        nn.init.constant_(self.user_bias.weight, 0.0)
        nn.init.constant_(self.item_bias.weight, 0.0)

        # DNN Layers
        for m in self.dnn.modules():
            if isinstance(m, nn.Linear):
                if self.init_method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def fm_interaction(self, u_emb, i_emb):
        # FM second-order interactions: 0.5 * ( (sum v)^2 - sum v^2 )
        # Here v is concatenation of user and item embeddings for each sample.
        x = torch.cat([u_emb, i_emb], dim=1)  # (batch, 2*emb_dim)
        sum_square = torch.sum(x, dim=1, keepdim=True)**2
        square_sum = torch.sum(x**2, dim=1, keepdim=True)
        return 0.5 * (sum_square - square_sum)

    def forward(self, user, item, feat):
        u_e = self.user_emb(user)  # (batch, emb)
        i_e = self.item_emb(item)  # (batch, emb)

        # biases
        u_b = self.user_bias(user)  # (batch,1)
        i_b = self.item_bias(item)  # (batch,1)

        if self.emb_dropout:
            u_e = self.emb_dropout(u_e)
            i_e = self.emb_dropout(i_e)

        # FM interaction
        fm = self.fm_interaction(u_e, i_e)  # (batch,1)

        # DNN on concatenated embeddings
        if feat is None:
            feat = torch.zeros(user.shape[0], self.num_features, device=u_e.device)
        concat = torch.cat([u_e, i_e, feat], dim=1)
        dnn_out = self.dnn(concat)  # (batch,1)

        out = u_b + i_b + fm + dnn_out  # (batch,1)
        return out.squeeze(1)


# -----------------------------
# Evaluation metrics
# -----------------------------

def hit_rate_at_k(recs, truth, k):
    return int(truth in recs[:k])


def ndcg_at_k(recs, truth, k):
    if truth in recs[:k]:
        rank = recs.index(truth)
        return 1.0 / math.log2(rank + 2)
    return 0.0


@torch.no_grad()
def evaluate_model(model, train_user_pos: dict, test_df: pd.DataFrame, num_items: int, item_feature_matrix: np.ndarray, k: int = 10, device: str = 'cpu'):
    model.to(device)
    model.eval()

    users = test_df['user_idx'].unique()
    HRs = []
    NDCGs = []

    # prepare item features tensor
    item_feat_tensor = torch.tensor(item_feature_matrix, dtype=torch.float, device=device)

    for u in users:
        # truth (single left-out item)
        truth = int(test_df[test_df['user_idx'] == u]['item_idx'].iloc[0])
        seen = train_user_pos.get(u, set())

        # candidates: all items excluding seen
        candidates = [i for i in range(num_items) if i not in seen]
        if len(candidates) == 0:
            HRs.append(0)
            NDCGs.append(0)
            continue

        user_tensor = torch.tensor([u] * len(candidates), dtype=torch.long, device=device)
        item_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        feat = item_feat_tensor[item_tensor]  # (num_cand, num_features)
        scores = model(user_tensor, item_tensor, feat).cpu().numpy()

        ranked_idx = np.argsort(-scores)
        recs = [candidates[i] for i in ranked_idx.tolist()]

        HRs.append(hit_rate_at_k(recs, truth, k))
        NDCGs.append(ndcg_at_k(recs, truth, k))

    return float(np.mean(HRs)), float(np.mean(NDCGs))


# -----------------------------
# Training loop with dynamic negatives + early stopping + checkpoint
# -----------------------------

def train_with_dynamic_negatives(feature_cols, item_feature_matrix, model: nn.Module,
                                 train_pos_df: pd.DataFrame,
                                 test_df: pd.DataFrame,
                                 num_users: int,
                                 num_items: int,
                                 epochs: int = 20,
                                 batch_size: int = 1024,
                                 lr: float = 1e-3,
                                 weight_decay: float = 1e-5,
                                 num_neg: int = 4,
                                 device: str = 'cpu',
                                 patience: int = 5,
                                 checkpoint_dir: str = 'checkpoints'):

    ensure_dir(checkpoint_dir)
    sampler = DynamicNegativeSampler(train_pos_df, num_items, num_neg=num_neg, feature_cols=feature_cols)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=False)
    loss_fn = nn.BCEWithLogitsLoss()

    best_ndcg = -1.0
    best_epoch = 0
    history = []

    # precompute user_pos map for filtering
    train_user_pos_map = train_pos_df[train_pos_df['label'] == 1].groupby('user_idx')['item_idx'].apply(set).to_dict()

    for ep in range(1, epochs + 1):
        t0 = time.time()
        # regenerate negatives dynamically each epoch
        train_aug = sampler.resample()
        dataset = RecDataset(train_aug, feature_cols)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        model.to(device)
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            users = batch['user'].to(device)
            items = batch['item'].to(device)
            labels = batch['label'].to(device)
            feats = batch['feat'].to(device)

            preds = model(users, items, feats)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)

        # evaluation (use fixed train_user_pos for filtering)
        hr, ndcg = evaluate_model(model, train_user_pos_map, test_df, num_items, item_feature_matrix, k=10, device=device)
        scheduler.step(1.0 - ndcg)  # Reduce LR when metric plateaus

        history.append({'epoch': ep, 'loss': avg_loss, 'hr': hr, 'ndcg': ndcg, 'time': time.time() - t0})

        print(f"Epoch {ep:02d} | loss={avg_loss:.4f} | HR@10={hr:.4f} | NDCG@10={ndcg:.4f} | time={history[-1]['time']:.1f}s")

        # checkpoint
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_epoch = ep
            ckpt_path = os.path.join(checkpoint_dir, f"best_model_epoch{ep}.pt")
            torch.save({'epoch': ep, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt_path)

        # early stopping
        if ep - best_epoch >= patience:
            print(f"Early stopping triggered (no improvement in {patience} epochs). Best epoch: {best_epoch}, best NDCG: {best_ndcg:.4f}")
            break

    return model, pd.DataFrame(history)


# -----------------------------
# Grid Search helper
# -----------------------------

def run_grid_search(feature_cols, item_feature_matrix, df, num_users, num_items, test_df, param_grid, out_csv='grid_search_results.csv', device='cpu'):
    rows = []
    total = 0
    # ensure mlp dims are tuples
    if 'mlp_dims' in param_grid:
        param_grid['mlp_dims'] = [tuple(v) for v in param_grid['mlp_dims']]

    for vals in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), vals))
        total += 1
        print(f"\n[Grid] Starting experiment {total} with params: {params}")

        seed_everything(params.get('seed', 42))
        model = DeepFM(len(feature_cols), num_users, num_items, emb_dim=params['embed_dim'], mlp_dims=tuple(params['mlp_dims']),
                       dropout=0.2,
                       use_batchnorm=True,
                       emb_dropout=0.05,
                       init_method='kaiming')

        model, history = train_with_dynamic_negatives(
            feature_cols,
            item_feature_matrix,
            model=model,
            train_pos_df=df,
            test_df=test_df,
            num_users=num_users,
            num_items=num_items,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            num_neg=params['num_neg'],
            device=device,
            patience=params.get('patience', 5)
        )

        train_user_pos_map = df[df['label'] == 1].groupby('user_idx')['item_idx'].apply(set).to_dict()
        hr, ndcg = evaluate_model(model, train_user_pos_map, test_df, num_items, item_feature_matrix, k=10, device=device)

        result = {**params, 'hr': hr, 'ndcg': ndcg}
        rows.append(result)
        df_rows = pd.DataFrame(rows)
        df_rows.to_csv(out_csv, index=False)
        print(f"[Grid] Result: HR@10={hr:.4f}, NDCG@10={ndcg:.4f} (saved to {out_csv})")

    return pd.DataFrame(rows)


# -----------------------------
# Main CLI
# -----------------------------

def main(args):
    seed_everything(args.seed)

    # load and preprocess
    raw_rating = load_movielens(args.ratings)
    raw_movie, genre_cols = load_movie(args.movies)
    train_pos_df, test_df, num_users, num_items, u2i, m2i = map_and_binarize(raw_rating, raw_movie, genre_cols, threshold=args.threshold)

    print(f"Users: {num_users}, Items: {num_items}, Train interactions: {len(train_pos_df)}, Test interactions: {len(test_df)}")

    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    feature_cols = genre_cols + ['year']

    # build item feature matrix (num_items x num_features) in item_idx order
    # We'll extract distinct item rows from train_pos_df (should include all items present in ratings)
    item_feat_df = train_pos_df[["item_idx"] + feature_cols].drop_duplicates("item_idx").set_index('item_idx')
    # for safety, ensure all item indices present
    missing = set(range(num_items)) - set(item_feat_df.index.tolist())
    if missing:
        # create zeros for missing
        for m in missing:
            item_feat_df.loc[m] = [0.0] * len(feature_cols)
    item_feat_df = item_feat_df.sort_index()
    item_feature_matrix = item_feat_df[feature_cols].fillna(0).values.astype(np.float32)

    if args.grid_search:
        param_grid = {
            'embed_dim': args.grid_embed_dim,
            'mlp_dims': args.grid_mlp_dims,
            'epochs': args.grid_epochs,
            'batch_size': args.grid_batch_size,
            'lr': args.grid_lr,
            'weight_decay': args.grid_weight_decay,
            'num_neg': args.grid_num_neg,
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
        results = run_grid_search(feature_cols, item_feature_matrix, train_pos_df, num_users, num_items, test_df, param_grid, out_csv=args.grid_out, device=device)
        print("Grid search finished. Results:\n", results)
        return

    # quick run
    model = DeepFM(len(feature_cols), num_users, num_items, emb_dim=args.embed_dim, mlp_dims=tuple(args.mlp_dims),
                   dropout=0.2,
                   use_batchnorm=True,
                   emb_dropout=0.05,
                   init_method='kaiming')
    model, history = train_with_dynamic_negatives(feature_cols, item_feature_matrix, model, train_pos_df, test_df, num_users, num_items,
                                                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                                   weight_decay=args.weight_decay, num_neg=args.num_neg,
                                                   device=device, patience=args.patience,
                                                   checkpoint_dir=args.checkpoint_dir)

    # final evaluation
    train_user_pos_map = train_pos_df[train_pos_df['label'] == 1].groupby('user_idx')['item_idx'].apply(set).to_dict()
    hr, ndcg = evaluate_model(model, train_user_pos_map, test_df, num_items, item_feature_matrix, k=args.K, device=device)
    print(f"Final HR@{args.K}: {hr:.4f} | NDCG@{args.K}: {ndcg:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings', type=str, default='ratings.csv')
    parser.add_argument('--movies', type=str, default='movies.csv')
    parser.add_argument('--threshold', type=float, default=4.0, help='binarization threshold (>=)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--mlp_dims', type=int, nargs='+', default=[64, 32])
    parser.add_argument('--num_neg', type=int, default=4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    # grid search args (accept lists)
    parser.add_argument('--grid_search', type=int, default=0)
    parser.add_argument('--grid_embed_dim', type=int, nargs='*', default=[24, 32, 48])
    parser.add_argument('--grid_mlp_dims', type=int, nargs='*', default=[[64,32], [128,64], [64,64]])
    parser.add_argument('--grid_epochs', type=int, nargs='*', default=[20])
    parser.add_argument('--grid_batch_size', type=int, nargs='*', default=[1024, 2048])
    parser.add_argument('--grid_lr', type=float, nargs='*', default=[5e-4, 1e-3, 2e-3])
    parser.add_argument('--grid_weight_decay', type=float, nargs='*', default=[1e-6, 1e-5, 5e-5])
    parser.add_argument('--grid_num_neg', type=int, nargs='*', default=[4, 8])
    parser.add_argument('--grid_seed', type=int, nargs='*', default=[42])
    parser.add_argument('--grid_out', type=str, default='grid_search_results.csv')

    # convenience quick mode
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'quick'])

    args = parser.parse_args()

    # adapt quick mode
    if args.mode == 'quick':
        args.epochs = 5
        args.batch_size = 2048

    main(args)
