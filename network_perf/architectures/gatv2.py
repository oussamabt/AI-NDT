"""GATv2 architecture implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from ..config import ArchitectureConfig


class GATv2Architecture(torch.nn.Module):
    """GATv2: Improved attention mechanism, often outperforms SAGE."""

    def __init__(self, num_node_features, config: ArchitectureConfig):
        super(GATv2Architecture, self).__init__()
        self.config = config
        hidden_channels = config.hidden_channels
        num_layers = config.num_layers
        heads = config.heads or 4
        dropout = config.dropout

        # Input processing
        self.input_norm = LayerNorm(num_node_features)

        # GATv2 layers with proper dimension handling
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Calculate per-head dimensions properly
        head_dim = hidden_channels // heads

        # First layer: input_features -> hidden_channels
        self.convs.append(GATv2Conv(
            num_node_features,
            head_dim,  # Per-head dimension
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            bias=False,
            share_weights=False  # Important for stability
        ))
        self.norms.append(LayerNorm(hidden_channels))  # heads * head_dim = hidden_channels

        # Hidden layers: hidden_channels -> hidden_channels
        for layer_idx in range(num_layers - 2):
            self.convs.append(GATv2Conv(
                hidden_channels,
                head_dim,  # Per-head dimension
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                bias=False,
                share_weights=False
            ))
            self.norms.append(LayerNorm(hidden_channels))

        # Final layer: hidden_channels -> hidden_channels (single head for stability)
        if num_layers > 1:
            self.convs.append(GATv2Conv(
                hidden_channels,
                hidden_channels,  # Single head output
                heads=1,
                dropout=0.0,  # Reduce dropout in final layer
                add_self_loops=True,
                bias=False,
                share_weights=False
            ))
            self.norms.append(LayerNorm(hidden_channels))

        # Simplified output processing
        self.output_dropout = nn.Dropout(dropout * 0.5)  # Reduce output dropout

        # Simpler prediction heads
        self.rtt_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, 1)
        )

        self.retrans_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, 1)
        )

        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Better weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Improved weight initialization for GATv2."""
        if isinstance(module, nn.Linear):
            # Use smaller initialization for stability
            torch.nn.init.xavier_uniform_(module.weight, gain=0.3)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, GATv2Conv):
            # Initialize attention parameters
            if hasattr(module, 'lin_l') and hasattr(module.lin_l, 'weight'):
                torch.nn.init.xavier_uniform_(module.lin_l.weight, gain=0.3)
            if hasattr(module, 'lin_r') and hasattr(module.lin_r, 'weight'):
                torch.nn.init.xavier_uniform_(module.lin_r.weight, gain=0.3)

    def forward(self, x, edge_index, edge_attr=None):
        # Input normalization
        if x.size(0) > 1:
            x = self.input_norm(x)

        # Apply GATv2 layers with careful dimension tracking
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Store residual for potential skip connection
            x_residual = x if x.size(-1) == self.hidden_channels else None

            # Apply convolution
            x = conv(x, edge_index)

            # Apply normalization
            if x.size(0) > 1:
                x = norm(x)

            # Add residual connection (only for hidden layers with matching dimensions)
            if x_residual is not None and i > 0 and x.size(-1) == x_residual.size(-1):
                x = x + x_residual * 0.1

            # Apply activation and dropout (except for last layer)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Final processing
        x = self.output_dropout(x)

        # Generate predictions
        rtt_pred = self.rtt_head(x)
        retrans_pred = self.retrans_head(x)

        return torch.cat([rtt_pred, retrans_pred], dim=1)