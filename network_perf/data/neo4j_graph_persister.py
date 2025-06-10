"""
Save AI-NDT Knowledge Graphs to Neo4j
=====================================

This module saves the three PyTorch Geometric knowledge graphs to Neo4j
with proper node types, relationships, and properties.
"""

import torch
import numpy as np
from neo4j import GraphDatabase
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import json
import os

# Define Neo4j credentials - keep them in this file
NEO4J_URI = "neo4j+s://010a6efb.databases.neo4j.io"
NEO4J_USER = "neo4j" 
NEO4J_PASSWORD = "E-0I75343IXIRlcFw3V5h7Ftoa1VrAYCKo9WuMdq2Og"

def get_neo4j_credentials() -> Tuple[str, str, str]:
    """
    Returns Neo4j connection credentials stored in this module
    
    Returns:
        tuple: (uri, username, password)
    """
    return NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class Neo4jKnowledgeGraphSaver:
    """
    Saves AI-NDT knowledge graphs from PyTorch Geometric format to Neo4j
    """

    def __init__(self, uri, user, password):
        """
        Initialize Neo4j connection

        Args:
            uri: Neo4j database URI
            user: Username
            password: Password
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Neo4j connection successful!")
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            raise

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

    def save_all_graphs(self, graphs, clear_existing=True):
        """
        Save all three AI-NDT knowledge graphs to Neo4j

        Args:
            graphs: Dictionary with 'topology', 'state', 'application' graphs
            clear_existing: Whether to clear existing data first
        """

        print("Saving AI-NDT Knowledge Graphs to Neo4j...")
        print("=" * 50)

        if clear_existing:
            self._clear_database()

        # Save each graph with proper node types and relationships
        self._save_topology_graph(graphs['topology'])
        self._save_state_graph(graphs['state'])
        self._save_application_graph(graphs['application'])

        # Create cross-graph relationships
        self._create_cross_graph_relationships()

        # Add metadata
        self._add_metadata(graphs)

        print("All knowledge graphs saved to Neo4j successfully")

        # Print statistics
        self._print_neo4j_statistics()

    def _clear_database(self):
        """Clear all existing data from Neo4j"""
        print("Clearing existing Neo4j data...")

        with self.driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")

            # Delete any constraints or indexes (optional)
            # session.run("DROP CONSTRAINT ON (n:TopologyNode) ASSERT n.node_id IS UNIQUE")

        print("Database cleared")

    def _save_topology_graph(self, topology_graph):
        """Save Network Topology Knowledge Graph"""
        print("Saving Network Topology Knowledge Graph...")

        with self.driver.session() as session:
            # Create topology nodes
            for i in range(topology_graph.x.shape[0]):
                features = topology_graph.x[i].numpy()

                session.run("""
                    CREATE (n:NetworkNode:TopologyNode {
                        node_id: $node_id,
                        asn_normalized: $asn_normalized,
                        latitude: $latitude,
                        longitude: $longitude,
                        degree: $degree,
                        centrality: $centrality,
                        connectivity_score: $connectivity_score,
                        graph_type: 'topology',
                        created_at: datetime(),
                        last_updated: datetime()
                    })
                """, {
                    'node_id': int(i),
                    'asn_normalized': float(features[0]),
                    'latitude': float(features[1]),
                    'longitude': float(features[2]),
                    'degree': float(features[3]),
                    'centrality': float(features[4]),
                    'connectivity_score': float(features[5])
                })

            # Create topology relationships
            edge_index = topology_graph.edge_index.numpy()
            edges_created = 0

            for i in range(edge_index.shape[1]):
                source = int(edge_index[0, i])
                target = int(edge_index[1, i])

                # Skip self-loops and duplicate edges
                if source >= target:
                    continue

                # Calculate edge properties
                source_features = topology_graph.x[source].numpy()
                target_features = topology_graph.x[target].numpy()

                # Geographic distance
                lat_diff = source_features[1] - target_features[1]
                lon_diff = source_features[2] - target_features[2]
                geo_distance = float(np.sqrt(lat_diff**2 + lon_diff**2))

                # ASN similarity
                asn_similarity = float(1.0 - abs(source_features[0] - target_features[0]))

                session.run("""
                    MATCH (a:TopologyNode {node_id: $source})
                    MATCH (b:TopologyNode {node_id: $target})
                    CREATE (a)-[r:TOPOLOGY_CONNECTED {
                        connection_type: 'network_infrastructure',
                        geographic_distance: $geo_distance,
                        asn_similarity: $asn_similarity,
                        weight: $weight,
                        created_at: datetime()
                    }]->(b)
                """, {
                    'source': source,
                    'target': target,
                    'geo_distance': geo_distance,
                    'asn_similarity': asn_similarity,
                    'weight': float((asn_similarity + (1.0 - min(1.0, geo_distance))) / 2.0)
                })
                edges_created += 1

            print(f"Created {topology_graph.x.shape[0]} topology nodes, {edges_created} relationships")

    def _save_state_graph(self, state_graph):
        """Save Network State Knowledge Graph"""
        print("Saving Network State Knowledge Graph...")

        with self.driver.session() as session:
            # Create state nodes
            for i in range(state_graph.x.shape[0]):
                features = state_graph.x[i].numpy()
                targets = state_graph.y[i].numpy() if hasattr(state_graph, 'y') else np.zeros(3)

                session.run("""
                    CREATE (n:NetworkNode:StateNode {
                        node_id: $node_id,
                        rtt: $rtt,
                        jitter: $jitter,
                        packet_loss: $packet_loss,
                        bandwidth_utilization: $bandwidth_util,
                        load_factor: $load_factor,
                        quality_score: $quality_score,
                        stability: $stability,
                        rtt_prediction: $rtt_prediction,
                        quality_prediction: $quality_prediction,
                        stability_prediction: $stability_prediction,
                        graph_type: 'state',
                        timestamp: datetime(),
                        created_at: datetime()
                    })
                """, {
                    'node_id': int(i),
                    'rtt': float(features[0]),
                    'jitter': float(features[1]),
                    'packet_loss': float(features[2]),
                    'bandwidth_util': float(features[3]),
                    'load_factor': float(features[4]),
                    'quality_score': float(features[5]),
                    'stability': float(features[6]),
                    'rtt_prediction': float(targets[0]),
                    'quality_prediction': float(targets[1]),
                    'stability_prediction': float(targets[2])
                })

            # Create state relationships
            edge_index = state_graph.edge_index.numpy()
            edges_created = 0

            for i in range(edge_index.shape[1]):
                source = int(edge_index[0, i])
                target = int(edge_index[1, i])

                # Skip self-loops and duplicate edges
                if source >= target:
                    continue

                # Calculate performance correlation
                source_features = state_graph.x[source].numpy()
                target_features = state_graph.x[target].numpy()

                rtt_similarity = 1.0 - abs(source_features[0] - target_features[0])
                quality_similarity = 1.0 - abs(source_features[5] - target_features[5])
                correlation_strength = float((rtt_similarity + quality_similarity) / 2.0)

                session.run("""
                    MATCH (a:StateNode {node_id: $source})
                    MATCH (b:StateNode {node_id: $target})
                    CREATE (a)-[r:PERFORMANCE_CORRELATED {
                        correlation_type: 'performance_similarity',
                        correlation_strength: $correlation,
                        rtt_similarity: $rtt_sim,
                        quality_similarity: $quality_sim,
                        created_at: datetime()
                    }]->(b)
                """, {
                    'source': source,
                    'target': target,
                    'correlation': correlation_strength,
                    'rtt_sim': float(rtt_similarity),
                    'quality_sim': float(quality_similarity)
                })
                edges_created += 1

            print(f"Created {state_graph.x.shape[0]} state nodes, {edges_created} relationships")

    def _save_application_graph(self, app_graph):
        """Save Application State Knowledge Graph"""
        print("Saving Application State Knowledge Graph...")

        # Application type mapping
        app_types = ['web_browsing', 'video_streaming', 'voip', 'gaming', 'file_transfer']

        with self.driver.session() as session:
            # Create application nodes
            for i in range(app_graph.x.shape[0]):
                features = app_graph.x[i].numpy()
                targets = app_graph.y[i].numpy() if hasattr(app_graph, 'y') else np.zeros(2)

                # Determine application type
                app_type_idx = int(features[0] * (len(app_types) - 1))
                app_type = app_types[min(app_type_idx, len(app_types) - 1)]

                # Calculate base network node ID
                base_node_id = int(features[7] * 149)  # Assuming 150 base nodes

                session.run("""
                    CREATE (n:Application:AppStateNode {
                        app_id: $app_id,
                        base_node_id: $base_node_id,
                        app_type: $app_type,
                        app_type_normalized: $app_type_norm,
                        qoe_score: $qoe_score,
                        user_satisfaction: $user_satisfaction,
                        response_time_normalized: $response_time,
                        throughput_normalized: $throughput,
                        availability: $availability,
                        resource_usage: $resource_usage,
                        qoe_prediction: $qoe_prediction,
                        satisfaction_prediction: $satisfaction_prediction,
                        graph_type: 'application',
                        created_at: datetime(),
                        last_updated: datetime()
                    })
                """, {
                    'app_id': int(i),
                    'base_node_id': base_node_id,
                    'app_type': app_type,
                    'app_type_norm': float(features[0]),
                    'qoe_score': float(features[1]),
                    'user_satisfaction': float(features[2]),
                    'response_time': float(features[3]),
                    'throughput': float(features[4]),
                    'availability': float(features[5]),
                    'resource_usage': float(features[6]),
                    'qoe_prediction': float(targets[0]),
                    'satisfaction_prediction': float(targets[1])
                })

            # Create application relationships
            edge_index = app_graph.edge_index.numpy()
            edges_created = 0

            for i in range(edge_index.shape[1]):
                source = int(edge_index[0, i])
                target = int(edge_index[1, i])

                # Skip self-loops and duplicate edges
                if source >= target:
                    continue

                # Determine relationship type
                source_features = app_graph.x[source].numpy()
                target_features = app_graph.x[target].numpy()

                source_base_node = int(source_features[7] * 149)
                target_base_node = int(target_features[7] * 149)

                if source_base_node == target_base_node:
                    # Same base node - resource competition
                    relationship_type = "RESOURCE_COMPETITION"
                    resource_competition = float((source_features[6] + target_features[6]) / 2.0)

                    session.run("""
                        MATCH (a:AppStateNode {app_id: $source})
                        MATCH (b:AppStateNode {app_id: $target})
                        CREATE (a)-[r:RESOURCE_COMPETITION {
                            competition_type: 'same_node_resources',
                            competition_strength: $competition,
                            base_node_id: $base_node,
                            created_at: datetime()
                        }]->(b)
                    """, {
                        'source': source,
                        'target': target,
                        'competition': resource_competition,
                        'base_node': source_base_node
                    })
                else:
                    # Different base nodes - application similarity
                    qoe_similarity = 1.0 - abs(source_features[1] - target_features[1])
                    app_type_similarity = 1.0 - abs(source_features[0] - target_features[0])

                    session.run("""
                        MATCH (a:AppStateNode {app_id: $source})
                        MATCH (b:AppStateNode {app_id: $target})
                        CREATE (a)-[r:APPLICATION_SIMILARITY {
                            similarity_type: 'cross_node_apps',
                            qoe_similarity: $qoe_sim,
                            app_type_similarity: $app_type_sim,
                            overall_similarity: $overall_sim,
                            created_at: datetime()
                        }]->(b)
                    """, {
                        'source': source,
                        'target': target,
                        'qoe_sim': float(qoe_similarity),
                        'app_type_sim': float(app_type_similarity),
                        'overall_sim': float((qoe_similarity + app_type_similarity) / 2.0)
                    })

                edges_created += 1

            print(f"Created {app_graph.x.shape[0]} application nodes, {edges_created} relationships")

    def _create_cross_graph_relationships(self):
        """Create relationships between different knowledge graphs"""
        print("Creating cross-graph relationships...")

        with self.driver.session() as session:
            # Link topology nodes to state nodes (1:1 mapping)
            result = session.run("""
                MATCH (t:TopologyNode), (s:StateNode)
                WHERE t.node_id = s.node_id
                CREATE (t)-[r:HAS_CURRENT_STATE {
                    relationship_type: 'topology_to_state',
                    created_at: datetime()
                }]->(s)
                RETURN count(r) as relationships_created
            """)
            topo_state_links = result.single()['relationships_created']

            # Link state nodes to applications running on them
            result = session.run("""
                MATCH (s:StateNode), (a:AppStateNode)
                WHERE s.node_id = a.base_node_id
                CREATE (a)-[r:DEPENDS_ON_NETWORK_STATE {
                    relationship_type: 'application_to_state',
                    dependency_strength: 0.8 + rand() * 0.2,
                    created_at: datetime()
                }]->(s)
                RETURN count(r) as relationships_created
            """)
            app_state_links = result.single()['relationships_created']

            # Link topology to applications (indirect through state)
            result = session.run("""
                MATCH (t:TopologyNode)-[:HAS_CURRENT_STATE]->(s:StateNode)<-[:DEPENDS_ON_NETWORK_STATE]-(a:AppStateNode)
                CREATE (a)-[r:HOSTED_ON_NETWORK_NODE {
                    relationship_type: 'application_to_topology',
                    hosting_strength: 1.0,
                    created_at: datetime()
                }]->(t)
                RETURN count(r) as relationships_created
            """)
            app_topo_links = result.single()['relationships_created']

            print(f"Created {topo_state_links} topology↔state, {app_state_links} app→state, {app_topo_links} app→topology links")

    def _add_metadata(self, graphs):
        """Add metadata about the knowledge graphs"""
        print("Adding metadata...")

        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'total_graphs': len(graphs),
            'topology_nodes': int(graphs['topology'].x.shape[0]),
            'topology_edges': int(graphs['topology'].edge_index.shape[1]),
            'state_nodes': int(graphs['state'].x.shape[0]),
            'state_edges': int(graphs['state'].edge_index.shape[1]),
            'application_nodes': int(graphs['application'].x.shape[0]),
            'application_edges': int(graphs['application'].edge_index.shape[1]),
            'total_nodes': sum(int(g.x.shape[0]) for g in graphs.values()),
            'total_edges': sum(int(g.edge_index.shape[1]) for g in graphs.values()),
            'graph_version': '1.0',
            'ai_ndt_version': 'prototype'
        }

        with self.driver.session() as session:
            session.run("""
                CREATE (m:Metadata:AIKDTMetadata {
                    creation_timestamp: $timestamp,
                    total_graphs: $total_graphs,
                    topology_nodes: $topology_nodes,
                    topology_edges: $topology_edges,
                    state_nodes: $state_nodes,
                    state_edges: $state_edges,
                    application_nodes: $application_nodes,
                    application_edges: $application_edges,
                    total_nodes: $total_nodes,
                    total_edges: $total_edges,
                    graph_version: $graph_version,
                    ai_ndt_version: $ai_ndt_version,
                    data_source: 'RIPE_Atlas',
                    created_at: datetime()
                })
            """, metadata)

        print("Metadata added")

    def _print_neo4j_statistics(self):
        """Print statistics of saved graphs in Neo4j"""
        print("\nNeo4j Knowledge Graph Statistics:")
        print("=" * 45)

        with self.driver.session() as session:
            # Node counts by type
            result = session.run("""
                MATCH (n:TopologyNode) RETURN count(n) as count
            """)
            topology_count = result.single()['count']

            result = session.run("""
                MATCH (n:StateNode) RETURN count(n) as count
            """)
            state_count = result.single()['count']

            result = session.run("""
                MATCH (n:AppStateNode) RETURN count(n) as count
            """)
            app_count = result.single()['count']

            # Relationship counts
            result = session.run("""
                MATCH ()-[r:TOPOLOGY_CONNECTED]->() RETURN count(r) as count
            """)
            topo_rels = result.single()['count']

            result = session.run("""
                MATCH ()-[r:PERFORMANCE_CORRELATED]->() RETURN count(r) as count
            """)
            state_rels = result.single()['count']

            result = session.run("""
                MATCH ()-[r]->()
                WHERE type(r) IN ['RESOURCE_COMPETITION', 'APPLICATION_SIMILARITY']
                RETURN count(r) as count
            """)
            app_rels = result.single()['count']

            result = session.run("""
                MATCH ()-[r]->()
                WHERE type(r) IN ['HAS_CURRENT_STATE', 'DEPENDS_ON_NETWORK_STATE', 'HOSTED_ON_NETWORK_NODE']
                RETURN count(r) as count
            """)
            cross_rels = result.single()['count']

            print(f"Topology Graph: {topology_count:,} nodes, {topo_rels:,} relationships")
            print(f"State Graph: {state_count:,} nodes, {state_rels:,} relationships")
            print(f"Application Graph: {app_count:,} nodes, {app_rels:,} relationships")
            print(f"Cross-Graph Links: {cross_rels:,} relationships")
            print(f"Total: {topology_count + state_count + app_count:,} nodes, {topo_rels + state_rels + app_rels + cross_rels:,} relationships")


def save_ai_ndt_graphs_to_neo4j(graphs, neo4j_uri, neo4j_user, neo4j_password, clear_existing=True):
    """
    Convenience function to save AI-NDT knowledge graphs to Neo4j

    Args:
        graphs: Dictionary with 'topology', 'state', 'application' PyG graphs
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        clear_existing: Whether to clear existing data

    Returns:
        bool: True if successful, False otherwise
    """

    saver = None
    try:
        # Initialize saver
        saver = Neo4jKnowledgeGraphSaver(neo4j_uri, neo4j_user, neo4j_password)

        # Save all graphs
        saver.save_all_graphs(graphs, clear_existing=clear_existing)

        return True

    except Exception as e:
        print(f"Error saving to Neo4j: {e}")
        return False

    finally:
        if saver:
            saver.close()
