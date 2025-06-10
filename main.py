"""
Main script for training and evaluating GNN models for network performance prediction.
Author: Oussama Ben Taarit
Thesis: "An ML based network digital twin for QoE estimation in Software Defined Network"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os

from network_perf.factory import AdvancedArchitectureFactory
from network_perf.data.ripe_atlas import load_ripe_atlas_for_evaluation
from network_perf.evaluation.evaluator import ProperModelEvaluator


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train and evaluate GNN models for network performance prediction'
    )
    parser.add_argument('--model', type=str, default='SAGE',
                        choices=['SAGE', 'GATv2', 'GIN', 'GraphTransformer', 
                                'ResGatedGCN', 'ChebNet', 'GENConv'],
                        help='GNN architecture to use')
    parser.add_argument('--min-probes', type=int, default=100,
                        help='Minimum number of probes for RIPE Atlas data')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (fraction)')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Validation set size (fraction)')
    parser.add_argument('--compare-all', action='store_true',
                        help='Compare all architectures')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials per architecture')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU if available')
    
    return parser.parse_args()


def train_single_model(data, arch_name, config, device='cpu', 
                       test_size=0.2, val_size=0.2, epochs=200, patience=20):
    """Train and evaluate a single GNN architecture"""
    print(f"\nTraining {arch_name} model")
    print("=" * 50)
    
    evaluator = ProperModelEvaluator(data, test_size=test_size, val_size=val_size)
    
    # Create model
    model = AdvancedArchitectureFactory.create_model(
        arch_name, data.x.shape[1], config, device
    )
    
    # Train with validation
    print(f"Training on {data.x.shape[0]} nodes with {data.edge_index.shape[1]//2} edges")
    training_results = evaluator.train_with_validation(
        model, config, num_epochs=epochs, patience=patience
    )
    
    # Evaluate on test set
    test_metrics = evaluator.evaluate_final_performance(training_results['model'])
    
    # Detect overfitting
    overfitting_analysis = evaluator.detect_overfitting(training_results)
    
    print(f"Results:")
    print(f"   Test R²: {test_metrics['test_r2_overall']:.4f}")
    print(f"   RTT prediction R²: {test_metrics['test_r2_rtt']:.4f}")
    print(f"   Retransmission prediction R²: {test_metrics['test_r2_retrans']:.4f}")
    print(f"   Overfitting risk: {overfitting_analysis['overall_risk']}")
    
    result = {
        'architecture': arch_name,
        'test_metrics': test_metrics,
        'training_history': {
            'train_losses': training_results['train_losses'],
            'val_losses': training_results['val_losses'],
            'val_r2_scores': training_results['val_r2_scores'],
        },
        'overfitting_analysis': overfitting_analysis,
        'model': training_results['model'],
    }
    
    return result, evaluator


def compare_architectures(data, architectures, trials_per_arch=3, device='cpu',
                         test_size=0.2, val_size=0.2, epochs=200, patience=20):
    """Compare multiple GNN architectures"""
    print("\nARCHITECTURE COMPARISON")
    print("=" * 50)
    
    configs = AdvancedArchitectureFactory.get_default_configs()
    results = []
    
    for arch_name in architectures:
        print(f"\nTesting {arch_name}")
        print("-" * 50)
        
        arch_results = []
        
        for trial in range(trials_per_arch):
            print(f"   Trial {trial + 1}/{trials_per_arch}...")
            
            try:
                config = configs[arch_name]
                result, _ = train_single_model(
                    data, arch_name, config, device, 
                    test_size, val_size, epochs, patience
                )
                
                arch_results.append({
                    'architecture': arch_name,
                    'trial': trial,
                    'test_r2': result['test_metrics']['test_r2_overall'],
                    'val_r2': result['training_history']['val_r2_scores'][-1],
                    'overfitting_risk': result['overfitting_analysis']['overall_risk'],
                    'test_metrics': result['test_metrics'],
                    'model': result['model'],
                })
                
                print(f"   Test R²: {result['test_metrics']['test_r2_overall']:.4f}")
                
            except Exception as e:
                print(f"   Failed: {e}")
                continue
        
        if arch_results:
            avg_test_r2 = np.mean([r['test_r2'] for r in arch_results])
            avg_overfitting_risk = [r['overfitting_risk'] for r in arch_results]
            
            print(f"   {arch_name} Summary:")
            print(f"      Mean Test R²: {avg_test_r2:.4f}")
            print(f"      Overfitting Risks: {avg_overfitting_risk}")
            
            results.extend(arch_results)
    
    # Summarize results
    print("\nFINAL RANKINGS")
    print("=" * 50)
    
    # Group by architecture
    arch_performance = {}
    for result in results:
        arch = result['architecture']
        if arch not in arch_performance:
            arch_performance[arch] = []
        arch_performance[arch].append(result['test_r2'])
    
    # Sort by mean performance
    sorted_archs = sorted(
        arch_performance.items(),
        key=lambda x: np.mean(x[1]), 
        reverse=True
    )
    
    print("Architecture Rankings:")
    for i, (arch, scores) in enumerate(sorted_archs, 1):
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{i}. {arch:<15}: {mean_score:.4f} ± {std_score:.4f} R²")
    
    return results, sorted_archs


def plot_training_history(result, save_path=None):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    # Training/validation loss
    plt.subplot(1, 2, 1)
    plt.plot(result['training_history']['train_losses'], label='Train Loss')
    plt.plot(result['training_history']['val_losses'], label='Validation Loss')
    plt.title(f"{result['architecture']} - Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Validation R²
    plt.subplot(1, 2, 2)
    plt.plot(result['training_history']['val_r2_scores'], label='Validation R²', color='green')
    plt.axhline(y=result['test_metrics']['test_r2_overall'], color='red', 
                linestyle='--', label=f'Test R²: {result["test_metrics"]["test_r2_overall"]:.4f}')
    plt.title(f"{result['architecture']} - Validation R²")
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def save_results(results, output_dir):
    """Save results to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"results_{timestamp}.pt")
    
    # Remove model objects for saving
    results_to_save = []
    for result in results:
        result_copy = result.copy()
        if 'model' in result_copy:
            del result_copy['model']
        results_to_save.append(result_copy)
    
    torch.save(results_to_save, filename)
    print(f"Results saved to {filename}")


def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("NETWORK PERFORMANCE PREDICTION WITH GNN")
    print("=" * 50)
    print("Author: Oussama Ben Taarit")
    print("Thesis: \"An ML based network digital twin for QoE estimation in Software Defined Network\"")
    print("=" * 50)
    
    # Set up device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load RIPE Atlas data
    print("\nLoading RIPE Atlas network data...")
    data = load_ripe_atlas_for_evaluation(min_probes=args.min_probes)
    
    if args.compare_all:
        # Compare all architectures
        architectures = ['SAGE', 'GATv2', 'GIN', 'GraphTransformer', 
                         'ResGatedGCN', 'ChebNet', 'GENConv']
        
        results, rankings = compare_architectures(
            data, architectures, args.trials, device,
            args.test_size, args.val_size, args.epochs, args.patience
        )
        
        # Save results
        save_results(results, args.output_dir)
        
    else:
        # Train single model
        configs = AdvancedArchitectureFactory.get_default_configs()
        config = configs[args.model]
        
        result, evaluator = train_single_model(
            data, args.model, config, device,
            args.test_size, args.val_size, args.epochs, args.patience
        )
        
        # Plot training history
        plot_path = os.path.join(args.output_dir, f"{args.model}_training_history.png")
        plot_training_history(result, save_path=plot_path)
        
        # Save model
        model_path = os.path.join(args.output_dir, f"{args.model}_model.pt")
        torch.save(result['model'].state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()