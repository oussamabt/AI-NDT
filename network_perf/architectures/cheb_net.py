"""ChebNet architecture implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, LayerNorm
from ..config import ArchitectureConfig


class ChebNetArchitecture(torch.nn.Module):
    """ChebNet: Spectral graph convolution with Chebyshev polynomials."""

    def __init__(self, num_node_features, config: ArchitectureConfig):
        super(ChebNetArchitecture, self).__init__()
        self.config = config
        hidden_channels = config.hidden_channels
        num_layers = config.num_layers
        dropout = config.dropout
        K = config.special_params.get('K', 3) if config.special_params else 3  # Chebyshev order

        self.input_norm = LayerNorm(num_node_features)

        # Chebyshev layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(ChebConv(num_node_features, hidden_channels, K=K, bias=False))
        self.norms.append(LayerNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(ChebConv(hidden_channels, hidden_channels, K=K, bias=False))
            self.norms.append(LayerNorm(hidden_channels))

        # Output heads
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

        # Apply Chebyshev layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = norm(x)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Generate predictions
        rtt_pred = self.rtt_head(x)
        retrans_pred = self.retrans_head(x)

        return torch.cat([rtt_pred, retrans_pred], dim=1)