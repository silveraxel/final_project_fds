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
from torch_geometric.nn import GATv2Conv


from params import *




def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = []
        for col, encoder in encoders.items():
            # Special handling for AggregatedTagEncoder (doesn't use df columns)
            if isinstance(encoder, AggregatedTagEncoder):
                xs.append(encoder(df))
            else:
                xs.append(encoder(df[col]))
        x = torch.cat(xs, dim=-1)

    return x, mapping



def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)
    
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr



def load_edge_csv_tags(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, one_edge_per_row=False, **kwargs):
    
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


class AggregatedTagEncoder:
    def __init__(self, tags_path, movie_mapping, model_name='all-MiniLM-L6-v2', device=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.tags_path = tags_path
        self.movie_mapping = movie_mapping
    
    @torch.no_grad()
    def __call__(self, df):
        try:
            tags_df = pd.read_csv(self.tags_path)
            
            # Get unique tags and encode them
            unique_tags = tags_df['tag'].unique()
            print(f"Encoding {len(unique_tags)} unique tags for movie features...")
            
            tag_embeddings = self.model.encode(
                unique_tags,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.device
            ).cpu()
            
            tag_to_embedding = {tag: tag_embeddings[i] for i, tag in enumerate(unique_tags)}
            
            # Initialize empty embeddings for all movies
            num_movies = len(self.movie_mapping)
            embedding_dim = tag_embeddings.shape[1]
            movie_tag_features = torch.zeros(num_movies, embedding_dim)
            
            # Group tags by movie
            movie_tags = tags_df.groupby('movieId')['tag'].apply(list).to_dict()
            
            # Aggregate tag embeddings for each movie
            for movie_id, tags_list in movie_tags.items():
                if movie_id in self.movie_mapping:
                    movie_idx = self.movie_mapping[movie_id]
                    
                    # Average all tag embeddings for this movie
                    tag_embs = [tag_to_embedding[tag] for tag in tags_list]
                    if tag_embs:
                        movie_tag_features[movie_idx] = torch.stack(tag_embs).mean(dim=0)
            
            print(f"Created tag features for {len(movie_tags)} movies")
            print(f"Tag feature dimension: {embedding_dim}")
            
            return movie_tag_features
            
        except FileNotFoundError:
            print("Warning: tags.csv not found, returning zero features")
            # Return zero features if tags not available
            return torch.zeros(len(self.movie_mapping), 384)  # Default MiniLM dimension



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



class GNN(torch.nn.Module):
    #GNN with Jumping Knowledge connections and batch normalization
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

#TOBEFIXED
"""
class HeteroEdgeAwareGNN(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, dropout=0.0, use_bn=False, 
                 num_layers=3, edge_dim=384, heads=4):
        super().__init__()
        self.num_layers = num_layers
        
        # Create separate convolutions for each edge type
        self.convs = torch.nn.ModuleDict()
        
        for edge_type in metadata[1]:  # metadata[1] contains edge types
            src, rel, dst = edge_type
            
            # Different edge types might have different edge dimensions
            current_edge_dim = edge_dim if rel == 'tags' else None
            
            self.convs[f'{src}__{rel}__{dst}'] = torch.nn.ModuleList([
                GATv2Conv(
                    (-1, -1),
                    hidden_channels // heads if i < num_layers - 1 else hidden_channels,
                    heads=heads if i < num_layers - 1 else 1,
                    edge_dim=current_edge_dim,
                    add_self_loops=False,
                    concat=i < num_layers - 1
                ) for i in range(num_layers)
            ])
        
        self.dropout = torch.nn.Dropout(dropout)
        
        if use_bn:
            self.bns = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
            ])
        self.use_bn = use_bn
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Process each layer
        for layer_idx in range(self.num_layers):
            x_dict_new = {}
            
            # For each destination node type, aggregate messages
            for edge_type, edge_index in edge_index_dict.items():
                src_type, rel_type, dst_type = edge_type
                
                # Get edge attributes if available
                edge_attr = None
                if edge_attr_dict is not None and edge_type in edge_attr_dict:
                    edge_attr = edge_attr_dict[edge_type]
                
                # Get the appropriate convolution
                conv_key = f'{src_type}__{rel_type}__{dst_type}'
                conv = self.convs[conv_key][layer_idx]
                
                # Apply convolution
                src_x = x_dict[src_type]
                dst_x = x_dict[dst_type]
                
                out = conv((src_x, dst_x), edge_index, edge_attr=edge_attr)
                
                # Aggregate messages to destination
                if dst_type not in x_dict_new:
                    x_dict_new[dst_type] = []
                x_dict_new[dst_type].append(out)
            
            # Combine messages for each node type
            for node_type, messages in x_dict_new.items():
                x = torch.stack(messages).mean(dim=0)  # Average messages
                
                if self.use_bn:
                    x = self.bns[layer_idx](x)
                
                if layer_idx < self.num_layers - 1:
                    x = x.relu()
                    x = self.dropout(x)
                
                x_dict[node_type] = x
        
        return x_dict

"""

""""
class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, dropout=0.0):
        super().__init__()
        input_dim = hidden_channels * 2
        
        self.input_proj = torch.nn.Linear(input_dim, hidden_channels)
        
        # Residual block
        self.block1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.output = torch.nn.Linear(hidden_channels, 1)
    
    def forward(self, x_user, x_movie, edge_label_index):
        combined = torch.cat([x_user[edge_label_index[0]], 
                             x_movie[edge_label_index[1]]], dim=-1)
        x = self.input_proj(combined)
        x = x + self.block1(x)  # Residual connection
        return self.output(x).squeeze(-1)

"""

class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, dropout=DEFAULT_DROPOUT, num_layers=DEFAULT_NUM_MLP_LAYERS):
        """
        Args:
            hidden_channels: Hidden dimension
            dropout: Dropout rate
            num_layers: Number of MLP layers (minimum 1)
        """
        super().__init__()
        
        input_dim = hidden_channels * 2
        self.num_layers = max(1, num_layers)  # At least 1 layer
        
        layers = []
        
        # First layer: input projection
        layers.append(torch.nn.Linear(input_dim, hidden_channels))
        layers.append(torch.nn.BatchNorm1d(hidden_channels))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        
        # Hidden layers (if num_layers > 2)
        for i in range(1, self.num_layers - 1):
            # Progressively reduce dimensions
            in_dim = hidden_channels // (2 ** (i - 1))
            out_dim = hidden_channels // (2 ** i)
            
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(torch.nn.BatchNorm1d(out_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        
        # Output layer
        final_hidden_dim = hidden_channels // (2 ** (self.num_layers - 2)) if self.num_layers > 1 else hidden_channels
        layers.append(torch.nn.Linear(final_hidden_dim, 1))
        
        self.mlp = torch.nn.Sequential(*layers)
    
    def forward(self, x_user, x_movie, edge_label_index):
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        combined = torch.cat([edge_feat_user, edge_feat_movie], dim=-1)
        return self.mlp(combined).squeeze(-1)




class Model(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, metadata,
                 dropout=0.0, use_bn=DEFAULT_USE_BN, num_gnn_layers=DEFAULT_NUM_GNN_LAYERS, num_mlp_layers=DEFAULT_NUM_MLP_LAYERS, jk_mode=DEFAULT_JK_MODE):

        super().__init__()
        
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        
        # GNN with Jumping Knowledge
        self.gnn = GNN(
            hidden_channels, 
            dropout=dropout, 
            use_bn=use_bn, 
            num_layers=num_gnn_layers,
            jk_mode=jk_mode
        )
        self.gnn = to_hetero(self.gnn, metadata, aggr=AGGREGATION)
        
        self.classifier = Classifier(
            hidden_channels, 
            dropout=dropout,
            num_layers=num_mlp_layers
        )

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



#TOBEFIXED
"""
class Model(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, metadata, dropout=0.0, use_bn=False, 
                 num_gnn_layers=3, use_edge_attr=True):
        super().__init__()
        
        self.use_edge_attr = use_edge_attr
        
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        
        # Use custom heterogeneous GNN that supports edge attributes
        self.gnn = HeteroEdgeAwareGNN(
            hidden_channels=hidden_channels,
            metadata=metadata,
            dropout=dropout,
            use_bn=use_bn,
            num_layers=num_gnn_layers,
            edge_dim=384,  # Tag embedding dimension
            heads=4
        )
        
        self.classifier = Classifier(hidden_channels, dropout=dropout * 0.5)

    def forward(self, data):
        user_emb = self.user_emb(data['user'].n_id)
        
        x_dict = {
            'user': user_emb,
            'movie': data['movie'].x,
        }
        
        # Prepare edge attributes
        edge_attr_dict = {}
        if self.use_edge_attr:
            for edge_type in data.edge_types:
                if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                    edge_attr_dict[edge_type] = data[edge_type].edge_attr
        
        # Forward pass with edge attributes
        x_dict = self.gnn(x_dict, data.edge_index_dict, edge_attr_dict)
        
        pred = self.classifier(
            x_dict['user'],
            x_dict['movie'],
            data['user', 'rates', 'movie'].edge_label_index
        )
        return pred
"""





