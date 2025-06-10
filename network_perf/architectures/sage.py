"""Improved SAGE architecture implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class ImprovedSAGEArchitecture(torch.nn.Module):
    """Improved SAGE architecture with better initialization and regularization."""

    def __init__(self, num_node_features, hidden_channels=64, num_targets=2, dropout=0.3, num_layers=3):
        super(ImprovedSAGEArchitecture, self).__init__()
        self.num_node_features = num_node_features
        self.num_layers = num_layers

        # Improved input processing
        self.input_norm = nn.BatchNorm1d(num_node_features)
        self.input_dropout = nn.Dropout(dropout * 0.5)  # Less aggressive input dropout

        # Feature transformation with residual connection
        self.feature_transform = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU()
        )

        # SAGE layers with proper normalization
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(num_node_features, hidden_channels, normalize=True, bias=False))
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True, bias=False))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Improved output heads with proper scaling
        self.rtt_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

        self.retrans_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

        self.dropout = dropout

        # Learnable feature importance (but smaller impact)
        self.feature_weights = nn.Parameter(torch.ones(num_node_features) * 0.1)

        # Learnable output scaling
        self.rtt_scale = nn.Parameter(torch.tensor(1.0))
        self.retrans_scale = nn.Parameter(torch.tensor(1.0))
        self.rtt_bias = nn.Parameter(torch.tensor(0.0))
        self.retrans_bias = nn.Parameter(torch.tensor(0.0))

        # Better initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier initialization for better gradient flow
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, x, edge_index, edge_attr=None):
        # Handle feature dimension mismatch
        if x.size(1) != self.num_node_features:
            if x.size(1) < self.num_node_features:
                padding = torch.zeros(x.size(0), self.num_node_features - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :self.num_node_features]

        # Input processing with normalization
        if x.size(0) > 1:
            x = self.input_norm(x)
        x = self.input_dropout(x)

        # Subtle feature weighting (less aggressive than before)
        feature_weights_norm = torch.sigmoid(self.feature_weights)  # Use sigmoid instead of softmax
        x_weighted = x * (1.0 + feature_weights_norm * 0.5)  # Additive rather than multiplicative

        # Direct feature processing
        direct_features = self.feature_transform(x_weighted)

        # Graph feature processing with residual connections
        graph_x = x_weighted
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = graph_x if graph_x.size(-1) == conv.out_channels else None

            graph_x = conv(graph_x, edge_index)

            if graph_x.size(0) > 1:
                graph_x = norm(graph_x)

            graph_x = F.relu(graph_x)

            # Add residual connection if dimensions match
            if residual is not None and residual.size(-1) == graph_x.size(-1):
                graph_x = graph_x + residual * 0.1  # Weak residual

            if i < len(self.convs) - 1:  # Don't apply dropout to last layer
                graph_x = F.dropout(graph_x, p=self.dropout, training=self.training)

        # Combine features
        combined_features = torch.cat([direct_features, graph_x], dim=1)

        # Generate predictions with learnable scaling
        rtt_pred = self.rtt_head(combined_features)
        retrans_pred = self.retrans_head(combined_features)

        # Apply learnable scaling and bias
        rtt_pred = rtt_pred * torch.abs(self.rtt_scale) + self.rtt_bias
        retrans_pred = retrans_pred * torch.abs(self.retrans_scale) + self.retrans_bias

        return torch.cat([rtt_pred, retrans_pred], dim=1)