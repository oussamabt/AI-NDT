"""Graph Transformer architecture implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm
from ..config import ArchitectureConfig


class GraphTransformerArchitecture(torch.nn.Module):
    """Graph Transformer: Full attention mechanism for graphs."""

    def __init__(self, num_node_features, config: ArchitectureConfig):
        super(GraphTransformerArchitecture, self).__init__()
        self.config = config
        hidden_channels = config.hidden_channels
        num_layers = config.num_layers
        heads = config.heads or 4
        dropout = config.dropout

        self.input_norm = LayerNorm(num_node_features)

        # Transformer layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(TransformerConv(
            num_node_features,
            hidden_channels // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=None,
            bias=False
        ))
        self.norms.append(LayerNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(
                hidden_channels,
                hidden_channels // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=None,
                bias=False
            ))
            self.norms.append(LayerNorm(hidden_channels))

        # Final layer
        self.convs.append(TransformerConv(
            hidden_channels,
            hidden_channels,
            heads=1,
            dropout=dropout,
            edge_dim=None,
            bias=False
        ))
        self.norms.append(LayerNorm(hidden_channels))

        # Output heads with attention pooling
        self.attention_pool = nn.MultiheadAttention(
            hidden_channels, num_heads=2, dropout=dropout, batch_first=True
        )

        self.rtt_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

        self.retrans_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

        self.dropout = dropout
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, edge_attr=None):
        if x.size(0) > 1:
            x = self.input_norm(x)

        # Apply Transformer layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x if x.size(-1) == self.config.hidden_channels else None

            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = norm(x)

            # Residual connection
            if x_res is not None and i > 0:
                x = x + x_res * 0.1

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Generate predictions
        rtt_pred = self.rtt_head(x)
        retrans_pred = self.retrans_head(x)

        return torch.cat([rtt_pred, retrans_pred], dim=1)