"""Configuration classes for GNN architectures."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ArchitectureConfig:
    """Enhanced configuration for different architectures."""
    
    name: str
    architecture_class: str
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Architecture-specific parameters
    heads: Optional[int] = None          # For attention models
    aggr: str = 'mean'                   # Aggregation method
    normalize: bool = True               # Layer normalization
    residual: bool = True                # Residual connections
    edge_dim: Optional[int] = None       # Edge feature dimension
    special_params: Dict[str, Any] = field(default_factory=dict)  # Model-specific parameters