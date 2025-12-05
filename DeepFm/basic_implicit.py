import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------
# 1. Load dataset (solo user, movie, rating)
# -------------------------------------------------------
def load_data(ratings_path="ratings.csv",movies_path="movies.csv"):
    ratings = pd.read_csv(ratings_path, sep=",")

    # binarizzazione semplice: rating >= 3 → positivo
    ratings["label"] = (ratings["rating"] >= 3).astype(int)

    # mapping indici
    user2idx = {u: i for i, u in enumerate(ratings["userId"].unique())}
    movie2idx = {m: i for i, m in enumerate(ratings["movieId"].unique())}

    ratings["user_idx"] = ratings["userId"].map(user2idx)
    ratings["movie_idx"] = ratings["movieId"].map(movie2idx)

    return ratings, user2idx, movie2idx

# -------------------------------------------------------
# 2. Leave-One-Out
# -------------------------------------------------------
def leave_one_out_split(ratings):
    train_rows = []
    test_rows = []
    for uid, group in ratings.groupby("user_idx"):
        group = group.sort_values("timestamp")
        test_rows.append(group.iloc[-1].to_dict())
        train_rows += [row.to_dict() for _, row in group.iloc[:-1].iterrows()]

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)

# -------------------------------------------------------
# 3. Negative Sampling
# -------------------------------------------------------
def negative_sampling(train_df, num_movies, num_neg=4):
    neg_samples = []
    user_pos_movies = train_df.groupby('user_idx')["movie_idx"].apply(set).to_dict()

    for user in user_pos_movies:
        watched = user_pos_movies[user]
        all_movies = set(range(num_movies))
        candidates = list(all_movies - watched)

        if len(candidates) == 0:
            continue

        chosen = np.random.choice(candidates, size=min(num_neg, len(candidates)), replace=False)

        for m in chosen:
            neg_samples.append({
                "user_idx": user,
                "movie_idx": m,
                "label": 0
            })

    return pd.concat([train_df, pd.DataFrame(neg_samples)], ignore_index=True)

# -------------------------------------------------------
# 4. PyTorch Dataset
# -------------------------------------------------------
class RecDataset(Dataset):
    def __init__(self, df):
        self.users = df["user_idx"].values
        self.movies = df["movie_idx"].values
        self.labels = df["label"].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "user": torch.tensor(self.users[idx], dtype=torch.long),
            "movie": torch.tensor(self.movies[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

# -------------------------------------------------------
# 5. DeepFM MINIMALE
# -------------------------------------------------------
class DeepFM(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.movie_emb = nn.Embedding(num_movies, embed_dim)

        # Linear (FM first-order)
        self.linear = nn.Linear(embed_dim * 2, 1)

        # DNN (FM second-order)
        self.dnn = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, user, movie):
        u = self.user_emb(user)
        m = self.movie_emb(movie)
        x = torch.cat([u, m], dim=1)

        linear_out = self.linear(x)
        dnn_out = self.dnn(x)

        out = linear_out + dnn_out
        return self.sigmoid(out).squeeze()

# -------------------------------------------------------
# 6. TRAINING
# -------------------------------------------------------
def train_model(model, train_loader, epochs, lr, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            user = batch["user"].to(device)
            movie = batch["movie"].to(device)
            label = batch["label"].to(device)

            opt.zero_grad()
            pred = model(user, movie)
            pred = pred.clamp(1e-6, 1 - 1e-6)
            loss = loss_fn(pred, label)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss = {total_loss / len(train_loader):.4f}")

    return model

# -------------------------------------------------------
# 7. Top-K recommendation
# -------------------------------------------------------
def recommend_top_k(model, user_id, num_movies, train_df, K=10, device="cpu"):
    model.eval()

    seen = set(train_df[train_df["user_idx"] == user_id]["movie_idx"].values)
    candidates = [m for m in range(num_movies) if m not in seen]

    user_tensor = torch.tensor([user_id] * len(candidates), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(candidates, dtype=torch.long).to(device)

    with torch.no_grad():
        scores = model(user_tensor, movie_tensor).cpu().numpy()

    top_idx = np.argsort(scores)[-K:][::-1]
    return [candidates[i] for i in top_idx]

# -------------------------------------------------------
# 8. METRICHE
# -------------------------------------------------------
def hr(recs, truth):
    return int(truth in recs)

def ndcg(recs, truth):
    if truth in recs:
        rank = recs.index(truth)
        return 1 / np.log2(rank + 2)
    return 0

def evaluate(model, test_df, num_movies, train_df, K, device):
    HRs, NDCGs = [], []

    for user in test_df["user_idx"].unique():
        truth = test_df[test_df["user_idx"] == user]["movie_idx"].iloc[0]
        recs = recommend_top_k(model, user, num_movies, train_df, K, device)
        HRs.append(hr(recs, truth))
        NDCGs.append(ndcg(recs, truth))

    return np.mean(HRs), np.mean(NDCGs)

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main(args):
    seed_everything()

    ratings, map_users, map_movies = load_data(args.ratings,args.movies)
    train_df, test_df = leave_one_out_split(ratings)
    print("Train",train_df)
    print("Ratings",ratings)
    train_df = negative_sampling(train_df, len(map_movies), num_neg=args.num_neg)

    train_loader = DataLoader(RecDataset(train_df), batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepFM(len(map_users), len(map_movies), embed_dim=args.embed_dim)
    model = train_model(model, train_loader, args.epochs, args.lr, device)

    HR, NDCG = evaluate(model, test_df, len(map_movies), train_df, args.K, device)
    print(f"HR@{args.K}: {HR:.4f} | NDCG@{args.K}: {NDCG:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=str, default="ratings.csv")
    parser.add_argument("--movies", type=str, default="movies.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--num_neg", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=32)
    args = parser.parse_args()

    main(args)
