"""Factory for creating GNN architecture instances."""

import torch
from .config import ArchitectureConfig

class AdvancedArchitectureFactory:
    """Factory for creating GNN architecture instances with optimized configurations."""

    @classmethod
    def create_model(cls, architecture_name: str, num_node_features: int,
                    config: ArchitectureConfig, device: str = 'cpu'):
        """Create a model instance based on architecture name.
        
        Args:
            architecture_name: Name of the architecture
            num_node_features: Number of input node features
            config: Configuration for the architecture
            device: Device to place the model on ('cpu' or 'cuda')
            
        Returns:
            Instantiated model
        """
        device = torch.device(device)
        
        # Import architectures here to avoid circular imports
        from .architectures.sage import ImprovedSAGEArchitecture
        from .architectures.gatv2 import GATv2Architecture
        from .architectures.gin import GINArchitecture
        from .architectures.graph_transformer import GraphTransformerArchitecture
        from .architectures.res_gated_gcn import ResGatedGCNArchitecture
        from .architectures.cheb_net import ChebNetArchitecture
        from .architectures.gen_conv import GENConvArchitecture
        
        architectures = {
            'SAGE': ImprovedSAGEArchitecture,
            'GATv2': GATv2Architecture,
            'GIN': GINArchitecture,
            'GraphTransformer': GraphTransformerArchitecture,
            'ResGatedGCN': ResGatedGCNArchitecture,
            'ChebNet': ChebNetArchitecture,
            'GENConv': GENConvArchitecture,
        }

        if architecture_name not in architectures:
            raise ValueError(f"Unknown architecture: {architecture_name}")

        architecture_class = architectures[architecture_name]

        if architecture_name == 'SAGE':
            return architecture_class(
                num_node_features=num_node_features,
                hidden_channels=config.hidden_channels,
                num_layers=config.num_layers,
                dropout=config.dropout
            ).to(device)
        else:
            return architecture_class(num_node_features, config).to(device)

    @classmethod
    def sget_default_configs(cls):
        """Get optimized configurations for all architectures.
        
        Returns:
            Dictionary of default configs for each architecture
        """
        configs = {
            'SAGE': ArchitectureConfig(
                name='SAGE',
                architecture_class='SAGE',
                hidden_channels=128,
                num_layers=2,
                dropout=0.3,
                learning_rate=0.001
            ),
            'GATv2': ArchitectureConfig(
                name='GATv2',
                architecture_class='GATv2',
                hidden_channels=128,
                num_layers=2,
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=1e-5,
                heads=8
            ),
            'GIN': ArchitectureConfig(
                name='GIN',
                architecture_class='GIN',
                hidden_channels=128,
                num_layers=4,
                dropout=0.2,
                learning_rate=0.0005
            ),
            'GraphTransformer': ArchitectureConfig(
                name='GraphTransformer',
                architecture_class='GraphTransformer',
                hidden_channels=128,
                num_layers=3,
                dropout=0.1,
                learning_rate=0.0003,
                heads=8
            ),
            'ResGatedGCN': ArchitectureConfig(
                name='ResGatedGCN',
                architecture_class='ResGatedGCN',
                hidden_channels=256,
                num_layers=3,
                dropout=0.15,
                learning_rate=0.002,
                weight_decay=1e-5
            ),
            'ChebNet': ArchitectureConfig(
                name='ChebNet',
                architecture_class='ChebNet',
                hidden_channels=128,
                num_layers=3,
                dropout=0.3,
                learning_rate=0.001,
                special_params={'K': 3}
            ),
            'GENConv': ArchitectureConfig(
                name='GENConv',
                architecture_class='GENConv',
                hidden_channels=128,
                num_layers=3,
                dropout=0.2,
                learning_rate=0.0008
            )
        }
        
        return configs