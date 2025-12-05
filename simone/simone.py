import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import tempfile

from torch_geometric.nn import GCNConv  # puoi cambiare in GraphSAGEConv ecc.
from torch_geometric.data import Data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Carica MovieLens (movies/ratings/tags), mappa userId/movieId a indici, crea titoli per i nodi film.
Estrae feature per i film: tag più frequenti (multi‑hot) + generi (multi‑hot), concatena; utenti hanno feature zero.
Costruisce grafo bipartito utenti↔film dagli archi di rating (due direzioni) con edge_index PyG e Data.x dalle feature.
Modello GCN multi‑layer: produce embedding nodi, predice rating via prodotto elemento‑per‑elemento user/film e linear.
Train/val split, addestramento MSE con DataLoader; poi stampa raccomandazioni top‑k per un utente di esempio.

Nodi: utenti (indici 0..num_users-1) e film (indici num_users..num_users+num_movies-1).
Feature: 
    utenti = vettore di zeri; 
    film = concatenazione multi‑hot di tag più frequenti (top_k=200) + generi presenti in movies.csv.
Archi: 
    per ogni rating utente‑film viene creato un arco utente→film e film→utente (grafo bipartito non orientato); edge_index 2 x (2*num_ratings). 

'''

TOP_K_TAGS = 500  # usato se non si esegue lo sweep
TASK_TYPE = "regression"  # "regression" oppure "binary"
BINARY_THRESHOLD = 4.0    # usato solo se TASK_TYPE == "binary"
POSITIVE_TAG_THRESHOLD = 3.5  # rating minimo per considerare un tag "positivo"


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================
# 1. Caricamento dei dati
# ===========================

def load_movielens(movies_path, ratings_path, tags_path):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path)

    # Teniamo solo le colonne che ci servono
    ratings = ratings[["userId", "movieId", "rating"]]

    return movies, ratings, tags


def filter_positive_tags(tags, ratings, threshold=4.0):
    """
    Restituisce solo i tag associati a film che l'utente ha valutato con rating >= soglia.
    """
    if tags is None or tags.empty:
        return tags

    positive_pairs = ratings.loc[ratings["rating"] >= threshold, ["userId", "movieId"]]
    if positive_pairs.empty:
        return tags.iloc[0:0].copy()

    positive_pairs = positive_pairs.drop_duplicates()
    return tags.merge(positive_pairs, on=["userId", "movieId"], how="inner")


# ===========================
# 2. Mappe ID -> indici
# ===========================

def build_id_mappings(ratings, movies):
    """
    Mappa:
      userId originale  -> user_index [0 .. num_users-1]
      movieId originale -> movie_index [0 .. num_movies-1]
    Poi costruiamo:
      node_index per gli utenti    = user_index
      node_index per i film        = num_users + movie_index
    """
    unique_user_ids = ratings["userId"].unique()
    unique_movie_ids = ratings["movieId"].unique()

    user_id_to_index = {uid: i for i, uid in enumerate(unique_user_ids)}
    movie_id_to_index = {mid: i for i, mid in enumerate(unique_movie_ids)}

    num_users = len(user_id_to_index)
    num_movies = len(movie_id_to_index)
    num_nodes = num_users + num_movies

    # Dizionari inversi (ci servono per tornare ai raw id)
    index_to_user_id = {i: uid for uid, i in user_id_to_index.items()}
    index_to_movie_id = {i: mid for mid, i in movie_id_to_index.items()}

    # Per comodità, creiamo anche una mappa node_index -> info film
    movie_id_to_title = dict(zip(movies["movieId"], movies["title"]))
    node_index_to_movie_title = {}
    for raw_mid, movie_idx in movie_id_to_index.items():
        node_idx = num_users + movie_idx
        node_index_to_movie_title[node_idx] = movie_id_to_title.get(raw_mid, "Unknown")

    mappings = {
        "user_id_to_index": user_id_to_index,
        "movie_id_to_index": movie_id_to_index,
        "index_to_user_id": index_to_user_id,
        "index_to_movie_id": index_to_movie_id,
        "node_index_to_movie_title": node_index_to_movie_title,
        "num_users": num_users,
        "num_movies": num_movies,
        "num_nodes": num_nodes,
    }
    return mappings


# ===========================
# 3. Feature: tag/genre -> vettori film
# ===========================

def build_movie_tag_features(tags, mappings, top_k=200):
    """
    Crea una matrice [num_movies, num_tags] con feature multi-hot dei tag più frequenti.
    """
    tags = tags.dropna(subset=["tag"]).copy()
    tags["tag_norm"] = tags["tag"].str.lower().str.strip()
    tags = tags[tags["movieId"].isin(mappings["movie_id_to_index"])]

    top_tags = tags["tag_norm"].value_counts().head(top_k).index.tolist()
    num_movies = mappings["num_movies"]

    if len(top_tags) == 0:
        return torch.zeros((num_movies, 1), dtype=torch.float), {}

    tag_to_idx = {tag: i for i, tag in enumerate(top_tags)}
    movie_tag_matrix = torch.zeros((num_movies, len(top_tags)), dtype=torch.float)

    for _, row in tags.iterrows():
        tag = row["tag_norm"]
        col = tag_to_idx.get(tag)
        if col is None:
            continue
        movie_idx = mappings["movie_id_to_index"][row["movieId"]]
        movie_tag_matrix[movie_idx, col] = 1.0

    return movie_tag_matrix, tag_to_idx


def build_movie_genre_features(movies, mappings):
    """
    Crea una matrice [num_movies, num_genres] con feature multi-hot dei generi.
    """
    all_genres = []
    for g in movies["genres"].fillna("").astype(str):
        all_genres.extend(g.split("|"))
    unique_genres = sorted({g for g in all_genres if g and g != "(no genres listed)"})

    num_movies = mappings["num_movies"]
    if len(unique_genres) == 0:
        return torch.zeros((num_movies, 1), dtype=torch.float), {}

    genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
    movie_genre_matrix = torch.zeros((num_movies, len(unique_genres)), dtype=torch.float)

    for _, row in movies.iterrows():
        raw_mid = row["movieId"]
        if raw_mid not in mappings["movie_id_to_index"]:
            continue
        movie_idx = mappings["movie_id_to_index"][raw_mid]
        genres = str(row["genres"]).split("|")
        for g in genres:
            if g and g != "(no genres listed)":
                col = genre_to_idx.get(g)
                if col is not None:
                    movie_genre_matrix[movie_idx, col] = 1.0

    return movie_genre_matrix, genre_to_idx


def build_node_feature_matrix(movie_feature_matrix, mappings):
    """
    Costruisce la matrice feature per tutti i nodi (utenti = zeri, film = feature multi-hot).
    """
    num_users = mappings["num_users"]
    num_nodes = mappings["num_nodes"]
    feat_dim = movie_feature_matrix.size(1)

    node_features = torch.zeros((num_nodes, feat_dim), dtype=torch.float)
    node_features[num_users:, :] = movie_feature_matrix
    return node_features


def preprocess_ratings(rating_values, task_type="regression", threshold=4.0):
    """
    Se task_type == "binary", converte i rating in 0/1 usando la soglia.
    Altrimenti ritorna i rating così come sono.
    """
    if task_type == "binary":
        return (rating_values >= threshold).float()
    return rating_values


def make_node_features(movies, tags, mappings, include_tag_features=True, top_k_tags=TOP_K_TAGS):
    """
    Costruisce le feature dei nodi combinando opzionalmente tag e sempre i generi.
    """
    feature_blocks = []

    if include_tag_features:
        movie_tag_matrix, _ = build_movie_tag_features(tags, mappings, top_k=top_k_tags)
        feature_blocks.append(movie_tag_matrix)

    movie_genre_matrix, _ = build_movie_genre_features(movies, mappings)
    feature_blocks.append(movie_genre_matrix)

    movie_feature_matrix = torch.cat(feature_blocks, dim=1)
    node_features = build_node_feature_matrix(movie_feature_matrix, mappings)
    return node_features


# ===========================
# 4. Costruzione del grafo
# ===========================

def build_tag_edges(tags, mappings):
    """
    Crea archi user-movie basati sui tag (se l'utente ha taggato il film).
    """
    user_id_to_index = mappings["user_id_to_index"]
    movie_id_to_index = mappings["movie_id_to_index"]
    num_users = mappings["num_users"]

    user_indices = []
    movie_indices = []

    for _, row in tags.iterrows():
        raw_uid = row["userId"]
        raw_mid = row["movieId"]
        if raw_uid not in user_id_to_index or raw_mid not in movie_id_to_index:
            continue
        u_idx = user_id_to_index[raw_uid]
        m_idx = movie_id_to_index[raw_mid]
        movie_node_idx = num_users + m_idx
        user_indices.append(u_idx)
        movie_indices.append(movie_node_idx)

    if len(user_indices) == 0:
        return None, None

    return torch.tensor(user_indices, dtype=torch.long), torch.tensor(movie_indices, dtype=torch.long)


def build_graph(ratings, mappings, node_features=None, tags_df=None, include_tag_edges=False):
    """
    Costruiamo:
      - edge_index: archi user_node -> movie_node (bipartito)
      - edge_ratings: rating associato a ogni arco
      - tensors per training (user_node_idx, movie_node_idx, rating)
    """
    user_id_to_index = mappings["user_id_to_index"]
    movie_id_to_index = mappings["movie_id_to_index"]
    num_users = mappings["num_users"]

    user_indices = []
    movie_indices = []
    rating_values = []

    for _, row in ratings.iterrows():
        raw_uid = row["userId"]
        raw_mid = row["movieId"]
        r = float(row["rating"])

        u_idx = user_id_to_index[raw_uid]          # [0 .. num_users-1]
        m_idx = movie_id_to_index[raw_mid]         # [0 .. num_movies-1]
        movie_node_idx = num_users + m_idx         # shift per stare dopo gli utenti

        user_indices.append(u_idx)
        movie_indices.append(movie_node_idx)
        rating_values.append(r)

    user_indices = torch.tensor(user_indices, dtype=torch.long)
    movie_indices = torch.tensor(movie_indices, dtype=torch.long)
    rating_values = torch.tensor(rating_values, dtype=torch.float)

    # Archi dai tag (se richiesti)
    tag_user_indices = None
    tag_movie_indices = None
    if include_tag_edges and tags_df is not None:
        tag_user_indices, tag_movie_indices = build_tag_edges(tags_df, mappings)

    # edge_index: 2 x num_edges
    # Arco non orientato: aggiungiamo sia user->movie che movie->user
    edges_src_list = [user_indices, movie_indices]
    edges_dst_list = [movie_indices, user_indices]

    if tag_user_indices is not None and tag_movie_indices is not None:
        edges_src_list.extend([tag_user_indices, tag_movie_indices])
        edges_dst_list.extend([tag_movie_indices, tag_user_indices])

    edges_src = torch.cat(edges_src_list, dim=0)
    edges_dst = torch.cat(edges_dst_list, dim=0)

    edge_index = torch.stack([edges_src, edges_dst], dim=0)

    data = Data(
        edge_index=edge_index,
        num_nodes=mappings["num_nodes"]
    )
    if node_features is not None:
        data.x = node_features

    # Per il training useremo (user_node_idx, movie_node_idx, rating)
    return data, user_indices, movie_indices, rating_values


# ===========================
# 5. Modello GNN
# ===========================

class GNNRecommender(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        """
        Restituisce gli embedding per tutti i nodi partendo dalle feature in input.
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        return x  # [num_nodes, hidden_dim]

    def predict_pairs(self, node_embeddings, user_nodes, movie_nodes):
        """
        node_embeddings: [num_nodes, hidden_dim]
        user_nodes, movie_nodes: [batch_size]
        Restituisce i rating predetti: [batch_size]
        """
        user_emb = node_embeddings[user_nodes]    # [B, H]
        movie_emb = node_embeddings[movie_nodes]  # [B, H]

        # Similarità tipo dot product
        interaction = user_emb * movie_emb        # [B, H]
        # Passiamo da H -> 1
        out = self.fc(interaction).squeeze(-1)    # [B]

        # Opzionale: puoi forzare in range 0-5 con sigmoid*5
        # out = torch.sigmoid(out) * 5.0

        return out


# ===========================
# 6. Train / Val split
# ===========================

def train_val_split(user_nodes, movie_nodes, ratings, val_ratio=0.1):
    num_samples = user_nodes.size(0)
    perm = torch.randperm(num_samples)
    
    user_nodes = user_nodes[perm]
    movie_nodes = movie_nodes[perm]
    ratings = ratings[perm]

    split = int(num_samples * (1 - val_ratio))
    train_data = (user_nodes[:split], movie_nodes[:split], ratings[:split])
    val_data = (user_nodes[split:], movie_nodes[split:], ratings[split:])

    return train_data, val_data


# ===========================
# 7. Training loop
# ===========================

def train_model(model, data, train_data, val_data, epochs=10, batch_size=1024, lr=1e-3, task_type="regression"):
    train_users, train_movies, train_ratings = train_data
    val_users, val_movies, val_ratings = val_data

    if task_type == "binary":
        loss_fn = F.binary_cross_entropy_with_logits
        metric_name = "BCE"
    else:
        loss_fn = F.mse_loss
        metric_name = "MSE"

    train_dataset = TensorDataset(train_users, train_movies, train_ratings)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data = data.to(DEVICE)
    model = model.to(DEVICE)
    val_users = val_users.to(DEVICE)
    val_movies = val_movies.to(DEVICE)
    val_ratings = val_ratings.to(DEVICE)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_users, batch_movies, batch_r in train_loader:
            batch_users = batch_users.to(DEVICE)
            batch_movies = batch_movies.to(DEVICE)
            batch_r = batch_r.to(DEVICE)

            optimizer.zero_grad()

            node_emb = model(data.x, data.edge_index)

            preds = model.predict_pairs(node_emb, batch_users, batch_movies)
            loss = loss_fn(preds, batch_r)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_users.size(0)

        avg_train_loss = total_loss / train_users.size(0)

        # Validation
        model.eval()
        with torch.no_grad():
            node_emb = model(data.x, data.edge_index)
            val_preds = model.predict_pairs(node_emb, val_users, val_movies)
            val_loss = loss_fn(val_preds, val_ratings).item()

        print(f"Epoch {epoch:02d} | Train {metric_name}: {avg_train_loss:.4f} | Val {metric_name}: {val_loss:.4f}")

    return model



# ===========================
# 8. Raccomandazioni per un utente
# ===========================

def recommend_for_user(model, data, mappings, raw_user_id, k=10):
    """
    raw_user_id: userId come appare nel CSV
    Ritorna i top-k film suggeriti (titolo e punteggio).
    """
    user_id_to_index = mappings["user_id_to_index"]
    node_index_to_movie_title = mappings["node_index_to_movie_title"]
    num_users = mappings["num_users"]
    num_nodes = mappings["num_nodes"]

    if raw_user_id not in user_id_to_index:
        raise ValueError("User ID non presente nel dataset!")

    user_index = user_id_to_index[raw_user_id]  # indice user
    user_node = torch.tensor([user_index], dtype=torch.long, device=DEVICE)

    model.eval()
    data = data.to(DEVICE)
    with torch.no_grad():
        node_emb = model(data.x, data.edge_index)
        # Consideriamo tutti i nodi film: da num_users a num_nodes-1
        movie_nodes = torch.arange(num_users, num_nodes, dtype=torch.long, device=DEVICE)

        user_nodes_batch = user_node.repeat(movie_nodes.size(0))

        scores = model.predict_pairs(node_emb, user_nodes_batch, movie_nodes)
        if TASK_TYPE == "binary":
            scores = torch.sigmoid(scores)  # probabilità di rating alto
        # prendiamo i top-k (gli indici tornati sono posizioni in movie_nodes, non node_index)
        topk_scores, topk_positions = torch.topk(scores, k)
        topk_node_indices = movie_nodes[topk_positions]

    top_movies = []
    for score, node_idx in zip(topk_scores.cpu().tolist(), topk_node_indices.cpu().tolist()):
        title = node_index_to_movie_title.get(node_idx, f"Movie {node_idx}")
        top_movies.append((title, score))

    return top_movies


def run_experiment(
    name,
    movies,
    ratings,
    tags,
    mappings,
    include_tag_features=True,
    include_tag_edges=False,
    task_type="regression",
    binary_threshold=4.0,
    epochs=5,
    batch_size=2048,
    lr=1e-3,
):
    """
    Esegue un esperimento con una configurazione specifica e restituisce la val loss finale.
    """
    print(f"\n=== Esperimento: {name} ===")
    print(f"tag features: {include_tag_features} | tag edges: {include_tag_edges}")

    set_seed(42)
    node_features = make_node_features(
        movies,
        tags,
        mappings,
        include_tag_features=include_tag_features,
        top_k_tags=TOP_K_TAGS
    )

    data, user_nodes, movie_nodes, rating_values = build_graph(
        ratings,
        mappings,
        node_features,
        tags_df=tags,
        include_tag_edges=include_tag_edges
    )
    rating_values_proc = preprocess_ratings(rating_values, task_type=task_type, threshold=binary_threshold)

    train_data, val_data = train_val_split(user_nodes, movie_nodes, rating_values_proc, val_ratio=0.1)

    model = GNNRecommender(in_dim=node_features.size(1), hidden_dim=64, num_layers=2, dropout=0.2)
    model = train_model(model, data, train_data, val_data, epochs=epochs, batch_size=batch_size, lr=lr, task_type=task_type)

    val_users, val_movies, val_ratings = val_data
    with torch.no_grad():
        node_emb = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
        val_preds = model.predict_pairs(node_emb, val_users.to(DEVICE), val_movies.to(DEVICE))
        if task_type == "binary":
            val_loss = F.binary_cross_entropy_with_logits(val_preds, val_ratings.to(DEVICE)).item()
        else:
            val_loss = F.mse_loss(val_preds, val_ratings.to(DEVICE)).item()

    metric_name = "BCE" if task_type == "binary" else "MSE"
    print(f"[{name}] Val {metric_name}: {val_loss:.4f}")
    return val_loss


# ===========================
# 9. Main
# ===========================

def main():
    print("Carico dati...")
    movies, ratings, tags = load_movielens("dataset/movies.csv", "dataset/ratings.csv", "dataset/tags.csv")
    positive_tags = filter_positive_tags(tags, ratings, threshold=POSITIVE_TAG_THRESHOLD)
    print(f"Tag totali: {len(tags)} | tag positivi (rating >= {POSITIVE_TAG_THRESHOLD}): {len(positive_tags)}")

    task_type = TASK_TYPE
    print(f"Task: {task_type}")
    set_seed(42)

    print("Costruisco mapping ID...")
    mappings = build_id_mappings(ratings, movies)

    configs = [
        {"name": "tags_as_feature", "include_tag_features": True, "include_tag_edges": False},
    ]

    results = []
    for cfg in configs:
        val_loss = run_experiment(
            name=cfg["name"],
            movies=movies,
            ratings=ratings,
            tags=positive_tags,
            mappings=mappings,
            include_tag_features=cfg["include_tag_features"],
            include_tag_edges=cfg["include_tag_edges"],
            task_type=task_type,
            binary_threshold=BINARY_THRESHOLD,
            epochs=10,
            batch_size=2048,
            lr=1e-3,
        )
        results.append((cfg["name"], val_loss))

    metric_name = "BCE" if task_type == "binary" else "MSE"
    print("\n=== Riepilogo esperimenti ===")
    for name, val_loss in results:
        print(f"{name}: Val {metric_name} = {val_loss:.4f}")

    # Esempio: raccomandazioni per un utente reale del dataset
    # example_user_id = ratings["userId"].iloc[0]
    # print(f"\nRaccomandazioni per l'utente {example_user_id}:")
    # recs = recommend_for_user(model, data, mappings, raw_user_id=example_user_id, k=10)
    # for title, score in recs:
    #     print(f"- {title} (score: {score:.3f})")


if __name__ == "__main__":
    main()
