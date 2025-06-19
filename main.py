import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from config import Config
from feature_engineering.data_engineering import data_engineer_benchmark, span_data_2d, span_data_3d
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import pickle
import dgl
from scipy.io import loadmat
import yaml

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", default=str, choices=['hogrl', 'rgtan'],
                       help="Specify which method to use: hogrl or rgtan")
    parser.add_argument("--dataset", default='amazon', choices=['amazon', 'yelpchi'],
                       help="Specify dataset: amazon or yelpchi")
    parser.add_argument("--seed", default=42, type=int,
                       help="Random seed for reproducibility")
    
    method = vars(parser.parse_args())['method']

    # Configuration files for comparison models only
    if method == 'rgtan':
        yaml_file = "config/rgtan_cfg.yaml"
    elif method == 'hogrl':
        yaml_file = "config/hogrl_cfg.yaml"
    else:
        raise NotImplementedError(f"Method {method} not supported in this comparison. Use 'hogrl' or 'rgtan'.")

    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    
    # Override with command line arguments
    args['method'] = method
    args['dataset'] = vars(parser.parse_args())['dataset']
    args['seed'] = vars(parser.parse_args())['seed']
    
    # Set random seeds for reproducibility
    np.random.seed(args['seed'])
    
    return args


def load_comparison_data(dataset_name, test_size=0.2):
    """
    Load and prepare data for HOGRL vs RGTAN comparison
    Supports Amazon and YelpChi datasets
    """
    if dataset_name == 'amazon':
        data_path = "data/Amazon.mat"
        data = loadmat(data_path)
        features = data['features']
        labels = data['label'].flatten()
        adj_matrix = data['homo_adj'] if 'homo_adj' in data else data['net_upu']
        
    elif dataset_name == 'yelpchi':
        data_path = "data/YelpChi.mat"
        data = loadmat(data_path)
        features = data['features']
        labels = data['label'].flatten()
        adj_matrix = data['homo_adj'] if 'homo_adj' in data else data['net_rur']
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Use 'amazon' or 'yelpchi'.")
    
    # Create train/test split
    num_nodes = features.shape[0]
    indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, 
        shuffle=True, random_state=42
    )
    
    # Create DGL graph
    import scipy.sparse as sp
    if not sp.issparse(adj_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix)
    
    g = dgl.from_scipy(adj_matrix)
    g = dgl.add_self_loop(g)
    
    return features, labels, train_idx, test_idx, g


def run_hogrl(args, features, labels, train_idx, test_idx, g):
    """Run HOGRL model with optimized hyperparameters"""
    from methods.hogrl.hogrl_main import hogrl_main
    
    print(f"\n{'='*50}")
    print(f"Running HOGRL on {args['dataset'].upper()} dataset")
    print(f"{'='*50}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Features dimension: {features.shape[1]}")
    print(f"Graph nodes: {g.num_nodes()}, edges: {g.num_edges()}")
    
    # Update args with dataset-specific configurations
    hogrl_args = args.copy()
    hogrl_args.update({
        'features': features,
        'labels': labels,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'graph': g,
        'learning_rate': 0.001,
        'batch_size': 256,
        'embedding_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'high_order_hops': 3,
        'attention_heads': 8,
        'epochs': 200
    })
    
    results = hogrl_main(hogrl_args)
    return results


def run_rgtan(args, features, labels, train_idx, test_idx, g):
    """Run RGTAN model with optimized hyperparameters"""
    from methods.rgtan.rgtan_main import rgtan_main, loda_rgtan_data
    
    print(f"\n{'='*50}")
    print(f"Running RGTAN on {args['dataset'].upper()} dataset")
    print(f"{'='*50}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Features dimension: {features.shape[1]}")
    print(f"Graph nodes: {g.num_nodes()}, edges: {g.num_edges()}")
    
    # Load additional RGTAN-specific data
    feat_data, labels_rgtan, train_idx_rgtan, test_idx_rgtan, g_rgtan, cat_features, neigh_features = loda_rgtan_data(
        args['dataset'], args['test_size'])
    
    # Update args with dataset-specific configurations  
    rgtan_args = args.copy()
    rgtan_args.update({
        'learning_rate': 0.0015,
        'batch_size': 128,
        'embedding_dim': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'temporal_window': 7,
        'risk_threshold': 0.7,
        'epochs': 200,
        'nei_att_heads': {'amazon': 4, 'yelpchi': 6}
    })
    
    results = rgtan_main(
        feat_data, g_rgtan, train_idx_rgtan, test_idx_rgtan, labels_rgtan, 
        rgtan_args, cat_features, neigh_features, 
        nei_att_head=rgtan_args['nei_att_heads'][args['dataset']]
    )
    return results


def compare_models(hogrl_results, rgtan_results, dataset_name):
    """Compare and display results between HOGRL and RGTAN"""
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS - {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    print(f"{'Metric':<15} {'HOGRL':<12} {'RGTAN':<12} {'Improvement':<12}")
    print(f"{'-'*60}")
    
    metrics = ['auc', 'f1', 'precision', 'recall', 'ap']
    
    for metric in metrics:
        if metric in hogrl_results and metric in rgtan_results:
            hogrl_val = hogrl_results[metric]
            rgtan_val = rgtan_results[metric]
            improvement = ((hogrl_val - rgtan_val) / rgtan_val) * 100
            
            print(f"{metric.upper():<15} {hogrl_val:<12.4f} {rgtan_val:<12.4f} {improvement:+8.2f}%")
    
    print(f"{'-'*60}")
    
    # Training time comparison
    if 'training_time' in hogrl_results and 'training_time' in rgtan_results:
        print(f"Training Time (min):")
        print(f"  HOGRL: {hogrl_results['training_time']:.1f}")
        print(f"  RGTAN: {rgtan_results['training_time']:.1f}")
        time_improvement = ((rgtan_results['training_time'] - hogrl_results['training_time']) / rgtan_results['training_time']) * 100
        print(f"  HOGRL is {time_improvement:.1f}% faster")


def main(args):
    """Main comparison function"""
    print(f"Starting comparison between HOGRL and RGTAN models")
    print(f"Dataset: {args['dataset']}")
    print(f"Random seed: {args['seed']}")
    
    # Load data
    features, labels, train_idx, test_idx, g = load_comparison_data(
        args['dataset'], args.get('test_size', 0.2)
    )
    
    if args['method'] == 'hogrl':
        results = run_hogrl(args, features, labels, train_idx, test_idx, g)
        print(f"\nHOGRL Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
            
    elif args['method'] == 'rgtan':
        results = run_rgtan(args, features, labels, train_idx, test_idx, g)
        print(f"\nRGTAN Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
    
    else:
        # Run both models for direct comparison
        print("Running both models for comparison...")
        
        hogrl_args = args.copy()
        hogrl_args['method'] = 'hogrl'
        hogrl_results = run_hogrl(hogrl_args, features, labels, train_idx, test_idx, g)
        
        rgtan_args = args.copy()
        rgtan_args['method'] = 'rgtan'
        rgtan_results = run_rgtan(rgtan_args, features, labels, train_idx, test_idx, g)
        
        # Compare results
        compare_models(hogrl_results, rgtan_results, args['dataset'])
        
        # Save results for reproducibility
        results_file = f"results_{args['dataset']}_comparison_seed{args['seed']}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump({
                'hogrl': hogrl_results,
                'rgtan': rgtan_results,
                'dataset': args['dataset'],
                'seed': args['seed']
            }, f)
        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main(parse_args())