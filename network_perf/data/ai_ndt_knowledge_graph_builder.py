"""
build_simple_ai_ndt_graphs - Standalone Function
================================================

This function builds three AI-NDT knowledge graphs from RIPE Atlas data
without requiring any external dependencies like Neo4j.

Usage:
    graphs = build_simple_ai_ndt_graphs(our_ripe_data)
"""

import torch
import numpy as np
from torch_geometric.data import Data
import random
import sys
import os

# Add module directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Neo4j functionality from knowledge_graph_2
from knowledge_graph_2 import get_neo4j_credentials, save_ai_ndt_graphs_to_neo4j

def build_simple_ai_ndt_graphs(ripe_data, random_seed=42, verbose=True):
    """
    Build three AI-NDT knowledge graphs from RIPE Atlas data.

    Args:
        ripe_data: PyTorch Geometric Data object with:
                  - x: Node features [num_nodes, num_features]
                  - edge_index: Edge connectivity [2, num_edges]
                  - y: Optional targets [num_nodes, num_targets]
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress information

    Returns:
        dict: Three knowledge graphs
            - 'topology': Network Topology Knowledge Graph
            - 'state': Network State Knowledge Graph
            - 'application': Application State Knowledge Graph
    """

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if verbose:
        print("Building AI-NDT Knowledge Graphs")
        print("=" * 45)
        print(f"Input: {ripe_data.x.shape[0]} nodes, {ripe_data.edge_index.shape[1]} edges")

    graphs = {}

    # 1. Network Topology Knowledge Graph
    if verbose:
        print("   Building Network Topology Knowledge Graph...")
    graphs['topology'] = _build_topology_graph(ripe_data)

    # 2. Network State Knowledge Graph
    if verbose:
        print("   Building Network State Knowledge Graph...")
    graphs['state'] = _build_state_graph(ripe_data)

    # 3. Application State Knowledge Graph
    if verbose:
        print("   Building Application State Knowledge Graph...")
    graphs['application'] = _build_application_graph(ripe_data, verbose)

    # Print statistics
    if verbose:
        _print_graph_statistics(graphs)

    return graphs

def _build_topology_graph(ripe_data):
    """
    Build Network Topology Knowledge Graph.
    Focus: Network infrastructure, ASN relationships, geographic distribution.

    Features: [asn_normalized, latitude, longitude, degree, centrality, connectivity_score]
    """

    num_nodes = ripe_data.x.shape[0]
    topology_features = torch.zeros((num_nodes, 6))

    for i in range(num_nodes):
        original_features = ripe_data.x[i]

        # Extract topology-relevant features
        # Assuming RIPE data format: [rtt, jitter, packet_loss, asn, lat, lon, meas_count, degree, neighbors, bias]

        asn = float(original_features[3]) if original_features.shape[0] > 3 else 0.0
        latitude = float(original_features[4]) if original_features.shape[0] > 4 else 0.0
        longitude = float(original_features[5]) if original_features.shape[0] > 5 else 0.0
        degree = float(original_features[7]) if original_features.shape[0] > 7 else 1.0
        neighbor_count = float(original_features[8]) if original_features.shape[0] > 8 else 1.0

        # Calculate derived metrics
        centrality = min(1.0, degree / 20.0)  # Normalize degree to centrality
        connectivity_score = min(1.0, (degree + neighbor_count) / 30.0)

        # Normalize ASN (handle the fact it might already be normalized)
        if abs(asn) < 1.0:  # Already normalized
            asn_normalized = asn
        else:  # Raw ASN value
            asn_normalized = (asn % 10000) / 10000.0

        topology_features[i] = torch.tensor([
            asn_normalized,     # ASN [0, 1]
            latitude,           # Latitude [-1, 1] (assuming pre-normalized)
            longitude,          # Longitude [-1, 1] (assuming pre-normalized)
            degree / 20.0,      # Normalized degree [0, 1]
            centrality,         # Network centrality [0, 1]
            connectivity_score  # Connectivity importance [0, 1]
        ])

    # Create topology edges based on ASN similarity and geographic proximity
    edge_list = []

    for i in range(num_nodes):
        # Limit connections per node to avoid too dense graphs
        max_connections = min(15, num_nodes - 1)
        connected = 0

        for j in range(num_nodes):
            if i == j or connected >= max_connections:
                continue

            # Calculate connection probability
            asn_similarity = 1.0 - abs(topology_features[i, 0] - topology_features[j, 0])

            # Geographic proximity
            lat_diff = topology_features[i, 1] - topology_features[j, 1]
            lon_diff = topology_features[i, 2] - topology_features[j, 2]
            geo_distance = torch.sqrt(lat_diff**2 + lon_diff**2)
            geo_proximity = max(0.0, 1.0 - geo_distance)

            # Combined connection score
            connection_score = asn_similarity * 0.6 + geo_proximity * 0.4

            # Add edge if score is high enough
            if connection_score > 0.5:
                edge_list.extend([[i, j], [j, i]])
                connected += 1

    # Remove duplicates and create edge index
    if edge_list:
        edge_set = set()
        for edge in edge_list:
            edge_tuple = tuple(sorted(edge))
            edge_set.add(edge_tuple)

        final_edges = []
        for edge_tuple in edge_set:
            final_edges.extend([[edge_tuple[0], edge_tuple[1]], [edge_tuple[1], edge_tuple[0]]])

        edge_index = torch.tensor(final_edges).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=topology_features, edge_index=edge_index)

def _build_state_graph(ripe_data):
    """
    Build Network State Knowledge Graph.
    Focus: Current performance metrics, quality scores, stability.

    Features: [rtt, jitter, packet_loss, bandwidth_util, load_factor, quality_score, stability]
    Targets: [rtt_prediction, quality_prediction, stability_prediction]
    """

    num_nodes = ripe_data.x.shape[0]
    state_features = torch.zeros((num_nodes, 7))
    targets = torch.zeros((num_nodes, 3))

    for i in range(num_nodes):
        original_features = ripe_data.x[i]

        # Extract performance metrics
        rtt = float(original_features[0]) if original_features.shape[0] > 0 else 0.0
        jitter = float(original_features[1]) if original_features.shape[0] > 1 else 0.0
        packet_loss = float(original_features[2]) if original_features.shape[0] > 2 else 0.0
        measurement_count = float(original_features[6]) if original_features.shape[0] > 6 else 1.0

        # Calculate additional state metrics
        bandwidth_utilization = torch.rand(1).item() * 0.7 + 0.2  # 20-90%
        load_factor = min(1.0, (abs(rtt) + abs(packet_loss) + abs(jitter)) / 3.0)

        # Quality score calculation (higher is better)
        rtt_penalty = abs(rtt) * 0.4
        loss_penalty = abs(packet_loss) * 0.4
        jitter_penalty = abs(jitter) * 0.2
        quality_score = max(0.1, 1.0 - (rtt_penalty + loss_penalty + jitter_penalty))

        # Stability calculation (based on jitter and measurement consistency)
        stability = max(0.1, 1.0 - abs(jitter) * 0.5)
        if measurement_count > 0:
            stability *= min(1.0, measurement_count / 5.0)  # More measurements = more stable

        state_features[i] = torch.tensor([
            rtt,                    # RTT (normalized from RIPE data)
            jitter,                 # Jitter (normalized from RIPE data)
            packet_loss,            # Packet loss (normalized from RIPE data)
            bandwidth_utilization,  # Simulated bandwidth utilization [0.2, 0.9]
            load_factor,            # Network load factor [0, 1]
            quality_score,          # Overall quality score [0.1, 1]
            stability               # Network stability [0.1, 1]
        ])

        # Prediction targets
        targets[i] = torch.tensor([
            rtt,           # RTT prediction target
            quality_score, # Quality prediction target
            stability      # Stability prediction target
        ])

    # Create performance correlation edges
    edge_list = []

    for i in range(num_nodes):
        # Connect to nodes with similar performance characteristics
        connections = 0
        max_connections = min(10, num_nodes - 1)

        for j in range(num_nodes):
            if i == j or connections >= max_connections:
                continue

            # Calculate performance similarity
            rtt_sim = 1.0 - abs(state_features[i, 0] - state_features[j, 0])
            quality_sim = 1.0 - abs(state_features[i, 5] - state_features[j, 5])
            stability_sim = 1.0 - abs(state_features[i, 6] - state_features[j, 6])

            # Overall similarity
            similarity = (rtt_sim + quality_sim + stability_sim) / 3.0

            if similarity > 0.7:  # High similarity threshold
                edge_list.extend([[i, j], [j, i]])
                connections += 1

    # Create edge index
    if edge_list:
        edge_index = torch.tensor(edge_list).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=state_features, edge_index=edge_index, y=targets)

def _build_application_graph(ripe_data, verbose=False):
    """
    Build Application State Knowledge Graph.
    Focus: Application QoE, user satisfaction, resource usage.

    Features: [app_type, qoe_score, user_satisfaction, response_time, throughput, availability, resource_usage, base_node_id]
    Targets: [qoe_prediction, satisfaction_prediction]
    """

    num_base_nodes = ripe_data.x.shape[0]

    # Application types with their network sensitivity profiles
    app_types = {
        'web_browsing': {'rtt_weight': 0.5, 'loss_weight': 0.3, 'jitter_weight': 0.2},
        'video_streaming': {'rtt_weight': 0.2, 'loss_weight': 0.5, 'jitter_weight': 0.3},
        'voip': {'rtt_weight': 0.3, 'loss_weight': 0.2, 'jitter_weight': 0.5},
        'gaming': {'rtt_weight': 0.6, 'loss_weight': 0.2, 'jitter_weight': 0.2},
        'file_transfer': {'rtt_weight': 0.1, 'loss_weight': 0.8, 'jitter_weight': 0.1}
    }

    app_names = list(app_types.keys())
    applications = []

    # Generate applications for each network node
    for base_node_id in range(num_base_nodes):
        base_features = ripe_data.x[base_node_id]

        # Extract network performance
        rtt = float(base_features[0]) if base_features.shape[0] > 0 else 0.0
        jitter = float(base_features[1]) if base_features.shape[0] > 1 else 0.0
        packet_loss = float(base_features[2]) if base_features.shape[0] > 2 else 0.0

        # Number of applications on this node (1-3)
        num_apps = torch.randint(1, 4, (1,)).item()

        for app_idx in range(num_apps):
            # Select application type
            app_type_name = random.choice(app_names)
            app_type_idx = app_names.index(app_type_name)
            app_config = app_types[app_type_name]

            # Calculate QoE based on network performance and app sensitivity
            qoe_score = _calculate_qoe(app_config, rtt, jitter, packet_loss)

            # Calculate user satisfaction (correlated with QoE but with some noise)
            satisfaction_noise = torch.randn(1).item() * 0.1
            user_satisfaction = max(0.1, min(1.0, qoe_score + satisfaction_noise))

            # Calculate application-specific metrics
            response_time = _calculate_app_response_time(app_type_name, rtt)
            throughput = _calculate_app_throughput(app_type_name, packet_loss)
            availability = max(0.8, 1.0 - abs(packet_loss) * 2.0)
            resource_usage = torch.rand(1).item() * 0.6 + 0.3  # 30-90%

            applications.append({
                'base_node_id': base_node_id,
                'app_type_idx': app_type_idx,
                'app_type_name': app_type_name,
                'qoe_score': qoe_score,
                'user_satisfaction': user_satisfaction,
                'response_time': response_time,
                'throughput': throughput,
                'availability': availability,
                'resource_usage': resource_usage
            })

    num_app_nodes = len(applications)
    if verbose:
        print(f"      Created {num_app_nodes} applications from {num_base_nodes} network nodes")

    # Create application feature matrix
    app_features = torch.zeros((num_app_nodes, 8))
    targets = torch.zeros((num_app_nodes, 2))

    for i, app in enumerate(applications):
        app_features[i] = torch.tensor([
            app['app_type_idx'] / (len(app_names) - 1),  # Normalized app type [0, 1]
            app['qoe_score'],                            # QoE score [0, 1]
            app['user_satisfaction'],                    # User satisfaction [0, 1]
            min(1.0, app['response_time'] / 2000.0),    # Normalized response time [0, 1]
            app['throughput'] / 100.0,                  # Normalized throughput [0, 1]
            app['availability'],                        # Availability [0, 1]
            app['resource_usage'],                      # Resource usage [0, 1]
            app['base_node_id'] / max(1, num_base_nodes - 1)  # Normalized base node ID [0, 1]
        ])

        targets[i] = torch.tensor([
            app['qoe_score'],
            app['user_satisfaction']
        ])

    # Create application interaction edges
    edge_list = []

    # Group applications by base node for resource competition edges
    node_to_apps = {}
    for i, app in enumerate(applications):
        base_id = app['base_node_id']
        if base_id not in node_to_apps:
            node_to_apps[base_id] = []
        node_to_apps[base_id].append(i)

    # 1. Resource competition edges (same node)
    for base_id, app_indices in node_to_apps.items():
        for i in range(len(app_indices)):
            for j in range(i + 1, len(app_indices)):
                app_i_idx = app_indices[i]
                app_j_idx = app_indices[j]

                # Resource competition based on usage
                usage_i = applications[app_i_idx]['resource_usage']
                usage_j = applications[app_j_idx]['resource_usage']

                if usage_i + usage_j > 1.2:  # High resource competition
                    edge_list.extend([[app_i_idx, app_j_idx], [app_j_idx, app_i_idx]])

    # 2. Similar application type edges (cross-node)
    for i in range(num_app_nodes):
        connections = 0
        max_connections = min(8, num_app_nodes - 1)

        for j in range(i + 1, num_app_nodes):
            if connections >= max_connections:
                break

            app_i = applications[i]
            app_j = applications[j]

            # Skip if same base node (already handled above)
            if app_i['base_node_id'] == app_j['base_node_id']:
                continue

            # Connect similar applications with similar QoE
            if app_i['app_type_name'] == app_j['app_type_name']:
                qoe_similarity = 1.0 - abs(app_i['qoe_score'] - app_j['qoe_score'])
                if qoe_similarity > 0.8:
                    edge_list.extend([[i, j], [j, i]])
                    connections += 1

    # Create edge index
    if edge_list:
        # Remove duplicate edges
        edge_set = set(tuple(edge) for edge in edge_list)
        edge_list = [list(edge) for edge in edge_set]
        edge_index = torch.tensor(edge_list).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=app_features, edge_index=edge_index, y=targets)

def _calculate_qoe(app_config, rtt, jitter, packet_loss):
    """Calculate Quality of Experience for an application"""

    # Convert normalized values to approximate real values for QoE calculation
    rtt_ms = abs(rtt) * 200.0      # Scale to 0-200ms range
    jitter_ms = abs(jitter) * 50.0  # Scale to 0-50ms range
    loss_pct = abs(packet_loss) * 10.0  # Scale to 0-10% range

    # Calculate weighted impact based on application sensitivity
    rtt_impact = min(1.0, rtt_ms / 500.0) * app_config['rtt_weight']
    jitter_impact = min(1.0, jitter_ms / 100.0) * app_config['jitter_weight']
    loss_impact = min(1.0, loss_pct / 15.0) * app_config['loss_weight']

    # Calculate QoE (1.0 = excellent, 0.0 = poor)
    qoe = 1.0 - (rtt_impact + jitter_impact + loss_impact)

    return max(0.1, min(1.0, qoe))

def _calculate_app_response_time(app_type, base_rtt):
    """Calculate application-specific response time"""

    base_ms = abs(base_rtt) * 200.0  # Convert to milliseconds

    # Application-specific multipliers
    multipliers = {
        'web_browsing': 3.0,    # Web pages need multiple round trips
        'video_streaming': 1.5, # Buffering helps
        'voip': 1.0,           # Real-time, direct correlation
        'gaming': 1.2,         # Slightly higher due to processing
        'file_transfer': 0.8   # Less sensitive to latency
    }

    multiplier = multipliers.get(app_type, 1.0)
    base_response = base_ms * multiplier

    # Add some random variation
    noise = torch.distributions.Exponential(50.0).sample().item()

    return max(10.0, base_response + noise)

def _calculate_app_throughput(app_type, packet_loss):
    """Calculate application-specific throughput"""

    # Base throughput ranges by application type (Mbps)
    base_ranges = {
        'web_browsing': (2, 20),
        'video_streaming': (5, 50),
        'voip': (0.1, 2),
        'gaming': (0.5, 5),
        'file_transfer': (10, 100)
    }

    min_bw, max_bw = base_ranges.get(app_type, (1, 10))
    base_throughput = torch.rand(1).item() * (max_bw - min_bw) + min_bw

    # Reduce throughput based on packet loss
    loss_factor = max(0.1, 1.0 - abs(packet_loss) * 3.0)

    return base_throughput * loss_factor

def _print_graph_statistics(graphs):
    """Print comprehensive statistics for all knowledge graphs"""

    print("\nAI-NDT Knowledge Graph Statistics:")
    print("=" * 50)

    total_nodes = sum(data.x.shape[0] for data in graphs.values())
    total_edges = sum(data.edge_index.shape[1] for data in graphs.values())

    print(f"OVERALL: {total_nodes} total nodes, {total_edges} total edges\n")

    for name, data in graphs.items():
        print(f"{name.upper()} KNOWLEDGE GRAPH:")
        print(f"  Nodes: {data.x.shape[0]:,}")
        print(f"  Node features: {data.x.shape[1]}")
        print(f"  Edges: {data.edge_index.shape[1]:,}")

        if hasattr(data, 'y') and data.y is not None:
            print(f"  Targets: {data.y.shape[1]} dimensions")
            print(f"  Target range: [{torch.min(data.y):.3f}, {torch.max(data.y):.3f}]")

        # Edge density
        max_possible_edges = data.x.shape[0] * (data.x.shape[0] - 1)
        if max_possible_edges > 0:
            density = data.edge_index.shape[1] / max_possible_edges
            print(f"  Edge density: {density:.4f}")

        # Feature statistics
        print(f"  Feature ranges:")
        for i in range(min(4, data.x.shape[1])):
            feat_min = torch.min(data.x[:, i]).item()
            feat_max = torch.max(data.x[:, i]).item()
            feat_mean = torch.mean(data.x[:, i]).item()
            print(f"    Feature {i}: [{feat_min:.3f}, {feat_max:.3f}] (μ={feat_mean:.3f})")

        print()  # Empty line between graphs

    # Graph relationship summary
    print("GRAPH RELATIONSHIPS:")
    if 'topology' in graphs and 'application' in graphs:
        topo_nodes = graphs['topology'].x.shape[0]
        app_nodes = graphs['application'].x.shape[0]
        app_ratio = app_nodes / topo_nodes if topo_nodes > 0 else 0
        print(f"  Application/Network ratio: {app_ratio:.1f}:1")

    print("Knowledge graphs ready for AI-NDT GNN training!")

# Example usage and testing
if __name__ == "__main__":
    print("Testing build_simple_ai_ndt_graphs Function")
    print("=" * 50)

    # Create sample RIPE Atlas-like data
    num_nodes = 150
    num_features = 10  # [rtt, jitter, packet_loss, asn, lat, lon, meas_count, degree, neighbors, bias]

    # Generate realistic network features
    x = torch.zeros(num_nodes, num_features)

    for i in range(num_nodes):
        x[i] = torch.tensor([
            torch.randn(1).item() * 0.3,      # RTT (normalized)
            torch.randn(1).item() * 0.2,      # Jitter (normalized)
            torch.abs(torch.randn(1)).item() * 0.1,  # Packet loss (normalized)
            torch.rand(1).item(),             # ASN (normalized)
            torch.rand(1).item() * 2 - 1,     # Latitude [-1, 1]
            torch.rand(1).item() * 2 - 1,     # Longitude [-1, 1]
            torch.rand(1).item() * 10,        # Measurement count
            torch.rand(1).item() * 15,        # Degree
            torch.rand(1).item() * 10,        # Neighbor count
            1.0                               # Bias
        ])

    # Create some edges
    edge_list = []
    for i in range(num_nodes):
        for j in range(i + 1, min(i + 8, num_nodes)):
            if torch.rand(1).item() > 0.7:  # 30% connection probability
                edge_list.extend([[i, j], [j, i]])

    edge_index = torch.tensor(edge_list).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

    # Create sample targets
    y = torch.randn(num_nodes, 2)

    # Create sample RIPE data
    sample_ripe_data = Data(x=x, edge_index=edge_index, y=y)

    print(f"Sample RIPE data: {sample_ripe_data.x.shape[0]} nodes, {sample_ripe_data.edge_index.shape[1]} edges")

    # Build AI-NDT knowledge graphs
    graphs = build_simple_ai_ndt_graphs(sample_ripe_data)

    print(f"\nSUCCESS: Built {len(graphs)} AI-NDT knowledge graphs!")

    # Test individual graph access
    print(f"\nGraph Access Test:")
    print(f"  Topology: {graphs['topology'].x.shape}")
    print(f"  State: {graphs['state'].x.shape}")
    print(f"  Application: {graphs['application'].x.shape}")

    # Ask user if they want to save to Neo4j
    save_to_neo4j = input("\nDo you want to save these graphs to Neo4j? (y/n): ")
    
    if save_to_neo4j.lower() == 'y':
        print("\nSaving graphs to Neo4j...")
        
        # Get Neo4j credentials from knowledge_graph_2.py
        neo4j_uri, neo4j_user, neo4j_password = get_neo4j_credentials()
        
        # Save the graphs to Neo4j
        success = save_ai_ndt_graphs_to_neo4j(graphs, neo4j_uri, neo4j_user, neo4j_password)
        
        if success:
            print("Successfully saved graphs to Neo4j!")
        else:
            print("Failed to save graphs to Neo4j. Check connection and credentials.")
    else:
        print("Skipping Neo4j save.")

    print(f"\nFunction ready for use with your RIPE Atlas data!")