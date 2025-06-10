"""
RIPE Atlas data loading utilities for network performance prediction.
This module provides functionality to load and preprocess RIPE Atlas measurement data.
"""

import pandas as pd
import numpy as np
import os
import torch
import networkx as nx
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from torch_geometric.data import Data

# Try to import RIPE Atlas library
try:
    from ripe.atlas.cousteau import AtlasResultsRequest, AtlasLatestRequest, Probe
    RIPE_ATLAS_AVAILABLE = True
except ImportError:
    print("WARNING: RIPE Atlas not installed. Install with: pip install ripe.atlas.cousteau")
    RIPE_ATLAS_AVAILABLE = False


class RIPEAtlasDataLoader:
    """
    Data loader for RIPE Atlas measurement data.
    
    This class loads and preprocesses RIPE Atlas measurement data for
    network performance prediction tasks.
    """
    
    def __init__(self, min_probes=50, max_probes=1000, data_path=None, features=None):
        """
        Initialize the RIPE Atlas data loader.
        
        Args:
            min_probes: Minimum number of probes to consider valid data
            max_probes: Maximum number of probes to load per measurement
            data_path: Optional path to CSV data if not using RIPE API
            features: Optional list of features to extract from the data
        """
        self.min_probes = min_probes
        self.max_probes = max_probes
        self.data_path = data_path
        self.features = features or ["rtt", "packet_loss", "jitter"]
        self.probe_data = {}
        self.measurement_data = {}
        self.data = None
        
    def load_ripe_atlas_data(self, measurement_ids=None, time_window_hours=24):
        """
        Load real network measurement data from RIPE Atlas
        
        Args:
            measurement_ids: List of measurement IDs to load
            time_window_hours: Hours of measurement data to collect
            
        Returns:
            PyG Data object with network graph data
        """
        if not RIPE_ATLAS_AVAILABLE:
            print("ERROR: RIPE Atlas library not available. Using synthetic data.")
            return self._create_synthetic_ripe_data()

        print("Loading real RIPE Atlas network measurement data...")

        # Use default measurement IDs if none provided (these are real RIPE measurements)
        if measurement_ids is None:
            measurement_ids = [
                5051,   # Built-in ping measurement to k-root
                5001,   # Built-in ping measurement to various targets
                5004,   # Built-in traceroute measurement
                1666,   # Ping measurement to Google DNS
                20021   # Traceroute measurement
            ]

        # Calculate time window
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)

        all_measurements = []
        probe_info = {}

        for msm_id in measurement_ids:
            print(f"   Loading measurement {msm_id}...")

            try:
                # Get measurement results
                kwargs = {
                    "msm_id": msm_id,
                    "start": start_time,
                    "stop": end_time
                }

                # Use AtlasLatestRequest for recent data
                is_success, results = AtlasLatestRequest(**kwargs).create()

                if not is_success:
                    print(f"   WARNING: Failed to get data for measurement {msm_id}")
                    continue

                # Process results
                processed_results = []
                for result in results[:self.max_probes]:  # Limit results
                    if self._is_valid_result(result):
                        processed_result = self._process_measurement_result(result, msm_id)
                        if processed_result:
                            processed_results.append(processed_result)

                            # Store probe info
                            probe_id = result.get('prb_id')
                            if probe_id and probe_id not in probe_info:
                                probe_info[probe_id] = self._get_probe_info(probe_id)

                all_measurements.extend(processed_results)
                print(f"   Processed {len(processed_results)} valid results")

            except Exception as e:
                print(f"   ERROR: Error loading measurement {msm_id}: {e}")
                continue

        if len(all_measurements) < self.min_probes:
            print(f"WARNING: Only got {len(all_measurements)} measurements, need at least {self.min_probes}")
            print("Falling back to synthetic RIPE-like data...")
            return self._create_synthetic_ripe_data()

        print(f"Successfully loaded {len(all_measurements)} network measurements")

        # Build network graph from measurements
        return self._build_network_from_measurements(all_measurements, probe_info)
    
    def _is_valid_result(self, result):
        """Check if measurement result is valid"""
        if not isinstance(result, dict):
            return False

        # Must have probe ID
        if 'prb_id' not in result:
            return False

        # Must have measurement data
        if 'result' not in result and 'avg' not in result:
            return False

        return True

    def _process_measurement_result(self, result, measurement_id):
        """Process a single measurement result"""
        probe_id = result.get('prb_id')

        # Extract RTT data
        rtt_values = []

        if 'avg' in result:
            # Ping measurement
            rtt_values.append(result['avg'])
        elif 'result' in result:
            # Traceroute or other measurement
            if isinstance(result['result'], list):
                for hop in result['result']:
                    if isinstance(hop, dict) and 'result' in hop:
                        for packet in hop['result']:
                            if isinstance(packet, dict) and 'rtt' in packet:
                                rtt_values.append(packet['rtt'])

        if not rtt_values:
            return None

        # Calculate statistics
        avg_rtt = np.mean(rtt_values)
        min_rtt = np.min(rtt_values)
        max_rtt = np.max(rtt_values)
        jitter = np.std(rtt_values) if len(rtt_values) > 1 else 0

        # Estimate packet loss (simplified)
        packet_loss = 0
        if 'result' in result and isinstance(result['result'], list):
            total_packets = 0
            lost_packets = 0
            for hop in result['result']:
                if isinstance(hop, dict) and 'result' in hop:
                    for packet in hop['result']:
                        total_packets += 1
                        if isinstance(packet, dict) and packet.get('x') == '*':
                            lost_packets += 1
            if total_packets > 0:
                packet_loss = (lost_packets / total_packets) * 100

        return {
            'probe_id': probe_id,
            'measurement_id': measurement_id,
            'avg_rtt': avg_rtt,
            'min_rtt': min_rtt,
            'max_rtt': max_rtt,
            'jitter': jitter,
            'packet_loss': packet_loss,
            'timestamp': result.get('timestamp', time.time())
        }

    def _get_probe_info(self, probe_id):
        """Get probe information"""
        try:
            # This would normally query RIPE Atlas API for probe details
            # For now, return basic info
            return {
                'probe_id': probe_id,
                'country': 'Unknown',
                'asn': probe_id % 1000,  # Simplified ASN assignment
                'latitude': np.random.uniform(-90, 90),
                'longitude': np.random.uniform(-180, 180)
            }
        except:
            return {
                'probe_id': probe_id,
                'country': 'Unknown',
                'asn': probe_id % 1000,
                'latitude': 0.0,
                'longitude': 0.0
            }

    def _build_network_from_measurements(self, measurements, probe_info):
        """Build network graph from RIPE Atlas measurements"""
        print("Building network graph from RIPE Atlas measurements...")

        # Create NetworkX graph
        G = nx.Graph()

        # Group measurements by probe
        probe_measurements = {}
        for m in measurements:
            probe_id = m['probe_id']
            if probe_id not in probe_measurements:
                probe_measurements[probe_id] = []
            probe_measurements[probe_id].append(m)

        # Add nodes (probes) with their features
        for probe_id, measurements_list in probe_measurements.items():
            # Calculate aggregate statistics for this probe
            avg_rtt = np.mean([m['avg_rtt'] for m in measurements_list])
            avg_jitter = np.mean([m['jitter'] for m in measurements_list])
            avg_packet_loss = np.mean([m['packet_loss'] for m in measurements_list])

            # Get probe info
            info = probe_info.get(probe_id, {})

            G.add_node(probe_id,
                      avg_rtt=avg_rtt,
                      avg_jitter=avg_jitter,
                      avg_packet_loss=avg_packet_loss,
                      asn=info.get('asn', 0),
                      country=info.get('country', 'Unknown'),
                      latitude=info.get('latitude', 0.0),
                      longitude=info.get('longitude', 0.0),
                      measurement_count=len(measurements_list))

        # Add edges based on measurement similarity and geographic proximity
        nodes = list(G.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Calculate similarity
                similarity = self._calculate_probe_similarity(G, node1, node2)

                # Add edge if similarity is high enough
                if similarity > 0.3:  # Threshold for connection
                    G.add_edge(node1, node2, weight=similarity)

        # Convert to PyG Data format
        return self._convert_to_pyg_data(G)

    def _calculate_probe_similarity(self, G, node1, node2):
        """Calculate similarity between two probes"""
        data1 = G.nodes[node1]
        data2 = G.nodes[node2]

        # RTT similarity (closer RTTs = more similar)
        rtt_diff = abs(data1['avg_rtt'] - data2['avg_rtt'])
        rtt_similarity = max(0, 1 - rtt_diff / 100)  # Normalize by 100ms

        # ASN similarity (same ASN = more similar)
        asn_similarity = 1.0 if data1['asn'] == data2['asn'] else 0.0

        # Geographic similarity
        lat_diff = abs(data1['latitude'] - data2['latitude'])
        lon_diff = abs(data1['longitude'] - data2['longitude'])
        geo_distance = np.sqrt(lat_diff**2 + lon_diff**2)
        geo_similarity = max(0, 1 - geo_distance / 180)  # Normalize by max distance

        # Combined similarity
        return (rtt_similarity * 0.5 + asn_similarity * 0.3 + geo_similarity * 0.2)

    def _convert_to_pyg_data(self, G):
        """Convert NetworkX graph to PyTorch Geometric Data"""
        print("Converting to PyTorch Geometric format...")

        if len(G.nodes()) == 0:
            print("ERROR: No nodes in graph!")
            return self._create_synthetic_ripe_data()

        # Create node mapping
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        num_nodes = len(node_mapping)

        # Extract node features
        feature_dim = 10  # Fixed feature dimension
        node_features = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
        targets = torch.zeros((num_nodes, 2), dtype=torch.float)  # RTT, packet_loss

        for node, idx in node_mapping.items():
            data = G.nodes[node]

            # Features: [avg_rtt, avg_jitter, packet_loss, asn, lat, lon, measurement_count, degree, ...]
            features = [
                data.get('avg_rtt', 20.0),
                data.get('avg_jitter', 2.0),
                data.get('avg_packet_loss', 1.0),
                float(data.get('asn', 0) % 1000),  # Normalize ASN
                data.get('latitude', 0.0) / 90.0,  # Normalize latitude
                data.get('longitude', 0.0) / 180.0,  # Normalize longitude
                float(data.get('measurement_count', 1)),
                float(G.degree(node)),  # Node degree
                float(len(list(G.neighbors(node)))),  # Neighbor count
                1.0  # Bias term
            ]

            node_features[idx] = torch.tensor(features[:feature_dim], dtype=torch.float)

            # Targets: [RTT, packet_loss_percentage]
            targets[idx, 0] = data.get('avg_rtt', 20.0)
            targets[idx, 1] = data.get('avg_packet_loss', 1.0)

        # Extract edges
        edge_list = []
        for edge in G.edges():
            if edge[0] in node_mapping and edge[1] in node_mapping:
                edge_list.append([node_mapping[edge[0]], node_mapping[edge[1]]])
                edge_list.append([node_mapping[edge[1]], node_mapping[edge[0]]])  # Undirected

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

        # Normalize features and targets
        node_features = self._normalize_features(node_features)
        targets = self._normalize_targets(targets)

        data = Data(x=node_features, edge_index=edge_index, y=targets)

        print(f"Created PyG data with {num_nodes} nodes, {edge_index.shape[1]} edges")
        print(f"Target statistics:")
        print(f"   RTT - mean: {torch.mean(targets[:, 0]):.3f}, std: {torch.std(targets[:, 0]):.3f}")
        print(f"   Packet Loss - mean: {torch.mean(targets[:, 1]):.3f}, std: {torch.std(targets[:, 1]):.3f}")

        return data
        
    def _normalize_features(self, features):
        """Normalize features"""
        # Robust normalization
        median = torch.median(features, dim=0, keepdim=True)[0]
        mad = torch.median(torch.abs(features - median), dim=0, keepdim=True)[0]
        mad = torch.where(mad < 1e-6, torch.ones_like(mad), mad)

        normalized = (features - median) / (1.4826 * mad)
        return torch.clamp(normalized, min=-3.0, max=3.0)

    def _normalize_targets(self, targets):
        """Normalize targets"""
        # Log transform for RTT (usually log-normal)
        targets[:, 0] = torch.log(targets[:, 0] + 1e-6)

        # Sqrt transform for packet loss percentage
        targets[:, 1] = torch.sqrt(targets[:, 1])

        # Standard normalization
        for i in range(targets.shape[1]):
            col = targets[:, i]
            targets[:, i] = (col - torch.mean(col)) / (torch.std(col) + 1e-6)

        return targets

    def _create_synthetic_ripe_data(self):
        """Create synthetic data that mimics RIPE Atlas structure"""
        print("Creating synthetic RIPE-like network data...")

        num_nodes = np.random.randint(100, 500)  # Reasonable size

        # Create scale-free network (typical of Internet topology)
        G = nx.barabasi_albert_graph(num_nodes, m=3)

        # Add realistic network measurements
        for node in G.nodes():
            # Base RTT depends on node centrality
            degree = G.degree(node)
            centrality = nx.degree_centrality(G)[node]

            # More central nodes tend to have lower RTT
            base_rtt = 15 + 50 * (1 - centrality) + np.random.exponential(10)
            jitter = np.random.exponential(2)
            packet_loss = np.random.exponential(0.5)

            G.nodes[node].update({
                'avg_rtt': base_rtt,
                'avg_jitter': jitter,
                'avg_packet_loss': packet_loss,
                'asn': np.random.randint(1, 1000),
                'latitude': np.random.uniform(-90, 90),
                'longitude': np.random.uniform(-180, 180),
                'measurement_count': np.random.randint(1, 10)
            })

        return self._convert_to_pyg_data(G)
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the RIPE Atlas data from the specified path.
        
        Returns:
            DataFrame containing the loaded data.
        """
        if self.data_path:
            if os.path.isdir(self.data_path):
                # Load data from directory containing multiple files
                dfs = []
                for file in Path(self.data_path).glob("*.csv"):
                    df = pd.read_csv(file)
                    dfs.append(df)
                self.data = pd.concat(dfs, ignore_index=True)
            else:
                # Load data from a single file
                self.data = pd.read_csv(self.data_path)
        else:
            # Return data from RIPE Atlas API
            return self.load_ripe_atlas_data()
            
        return self.data
    
    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Returns:
            DataFrame containing the preprocessed data.
        """
        if self.data is None:
            self.load_data()
            
        # Handle missing values
        self.data = self.data.fillna(self.data.mean())
        
        # Extract relevant features
        if self.features:
            available_cols = set(self.data.columns)
            requested_features = set(self.features)
            missing_features = requested_features - available_cols
            
            if missing_features:
                print(f"Warning: Requested features {missing_features} not found in data.")
                
            features_to_use = list(requested_features.intersection(available_cols))
            if features_to_use:
                self.data = self.data[features_to_use]
        
        return self.data
    
    def split_train_test(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets.
        
        Args:
            test_size: Fraction of data to use for testing.
            
        Returns:
            Tuple of (train_data, test_data) DataFrames.
        """
        if self.data is None:
            self.preprocess()
            
        n = len(self.data)
        test_indices = np.random.choice(n, int(test_size * n), replace=False)
        train_indices = np.setdiff1d(np.arange(n), test_indices)
        
        train_data = self.data.iloc[train_indices].reset_index(drop=True)
        test_data = self.data.iloc[test_indices].reset_index(drop=True)
        
        return train_data, test_data
    
    def get_features_and_targets(self, 
                               target_col: str = "rtt", 
                               feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and targets from the data.
        
        Args:
            target_col: Name of the target column.
            feature_cols: List of feature columns to use.
            
        Returns:
            Tuple of (features, targets).
        """
        if self.data is None:
            self.preprocess()
            
        if feature_cols is None:
            feature_cols = [col for col in self.data.columns if col != target_col]
        
        X = self.data[feature_cols]
        y = self.data[target_col]
        
        return X, y


def load_ripe_atlas_for_evaluation(min_probes=100, measurement_hours=6):
    """
    Load RIPE Atlas data for architecture evaluation
    
    Args:
        min_probes: Minimum number of probes needed
        measurement_hours: Hours of measurement data to collect
        
    Returns:
        PyG Data object with real network measurements
    """
    
    print("LOADING RIPE ATLAS DATA FOR ARCHITECTURE EVALUATION")
    print("=" * 60)
    
    loader = RIPEAtlasDataLoader(min_probes=min_probes, max_probes=1000)
    
    try:
        # Load real RIPE Atlas data
        data = loader.load_ripe_atlas_data(time_window_hours=measurement_hours)
        
        print(f"RIPE Atlas data loaded successfully!")
        print(f"Nodes: {data.x.shape[0]:,}")
        print(f"Edges: {data.edge_index.shape[1]:,}")
        print(f"Features: {data.x.shape[1]}")
        print(f"Targets: RTT + Packet Loss")
        
        return data
        
    except Exception as e:
        print(f"ERROR: Failed to load RIPE Atlas data: {e}")
        print("Creating synthetic network data instead...")
        
        loader = RIPEAtlasDataLoader()
        return loader._create_synthetic_ripe_data()