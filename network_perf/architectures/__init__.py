"""GNN architecture implementations for network performance prediction."""

# Import all architectures
from .sage import ImprovedSAGEArchitecture
from .gatv2 import GATv2Architecture
from .gin import GINArchitecture
from .graph_transformer import GraphTransformerArchitecture
from .res_gated_gcn import ResGatedGCNArchitecture
from .cheb_net import ChebNetArchitecture
from .gen_conv import GENConvArchitecture

__all__ = [
    'ImprovedSAGEArchitecture',
    'GATv2Architecture',
    'GINArchitecture', 
    'GraphTransformerArchitecture',
    'ResGatedGCNArchitecture',
    'ChebNetArchitecture',
    'GENConvArchitecture',
]