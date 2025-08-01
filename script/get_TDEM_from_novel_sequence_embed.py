'''
Generate Target-Gene Expression (TE) predictions from protein sequence embeddings

Usage:
python get_TE_from_novel_sequence_embed.py \
    --input_embed_file /path/to/<prefix>_<PLM>_embeddings.tsv.gz \
    --save_dir /path/to/output \
    --prefix test \
    --cell A549 \
    --best_model_path /path/to/best_model.pt \
    --batch_size 1000

Then, the TE matrix will be saved as test_<cell>_TE_matrix.csv in the specified directory.
'''

import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from rdkit import Chem, RDLogger

from utils import *

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdForceField')
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def validate_input_file(filepath):
    """
    Validate input embedding file exists and has proper format
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input embedding file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, nrows=5, sep = '\t', compression = 'gzip', index_col = 0)
        required_columns = ['uniprot_id']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input file must contain required columns: {required_columns}")
        
        if df.empty:
            raise ValueError("Input file is empty")
            
        print(f"Input file validation successful. Sample shape: {df.shape}")
        
    except Exception as e:
        raise ValueError(f"Error reading input embedding file: {e}")


def validate_model_file(filepath):
    """
    Validate model file exists
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")


def load_gene_info(datadir, gene_space):
    """
    Load gene information based on the specified gene space
    """
    cmap_ginfo_path = os.path.join(datadir, 'CMAP_ginfo.csv')
    
    if not os.path.exists(cmap_ginfo_path):
        raise FileNotFoundError(f"CMAP gene info file not found: {cmap_ginfo_path}")
    
    try:
        cmap_ginfo = pd.read_csv(cmap_ginfo_path, index_col=0)
        
        # Mapping gene_space to required parameters
        gene_space_map = {
            'landmark': lambda: cmap_ginfo[cmap_ginfo.feature_space == 'landmark'].index.astype(str),
            'best_inferred': lambda: cmap_ginfo[cmap_ginfo.feature_space != 'inferred'].index.astype(str),
        }
        
        if gene_space not in gene_space_map:
            raise ValueError(f"Invalid gene_space: {gene_space}. Choose from: {list(gene_space_map.keys())}")
        
        gene_list = gene_space_map[gene_space]()
        print(f"Loaded {len(gene_list)} genes for gene space: {gene_space}")
        
        return gene_list, cmap_ginfo
        
    except Exception as e:
        raise ValueError(f"Error loading gene information: {e}")


def load_string_data(datadir, gene_list):
    """
    Load STRING PPI data and create graph data
    """
    string_path = os.path.join(datadir, 'string_coexpression_score_ENTREZ_mapped.csv')
    
    if not os.path.exists(string_path):
        raise FileNotFoundError(f"STRING data file not found: {string_path}")
    
    try:
        # Create entity to ID mapping
        ent2id = {gene: idx for idx, gene in enumerate(gene_list)}
        
        # Load STRING data
        string_df = pd.read_csv(string_path)
        string_df['ens1'] = string_df.ent1.astype(str).map(ent2id)
        string_df['ens2'] = string_df.ent2.astype(str).map(ent2id)
        string_df_f = string_df.loc[:, ['ens1', 'ens2']].dropna()
        
        print(f"Loaded {len(string_df_f)} PPI edges")
        
        return string_df_f, ent2id
        
    except Exception as e:
        raise ValueError(f"Error loading STRING data: {e}")


def create_graph_data_without_score(edge_data):
    """
    Create graph data structure from edge data
    This is a placeholder - implement according to your graph structure
    """
    # Convert to tensor format for PyTorch Geometric or similar
    edges = torch.tensor(edge_data.values, dtype=torch.long).t().contiguous()
    return edges


class TargetToExpressionModel(nn.Module):
    """
    Placeholder model class - implement according to your architecture
    """
    def __init__(self, input_dim, gene_dim, hidden_dims, graph_node_embed_dim, gat_num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.gene_dim = gene_dim
        
        # Build layers based on hidden_dims
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, gene_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def generate_TE(args):
    """
    Generate Target-Expression model
    """
    return TargetToExpressionModel(
        input_dim=args.target_embed_dim,
        gene_dim=args.gene_dim,
        hidden_dims=args.hidden_dims,
        graph_node_embed_dim=args.graph_node_embed_dim,
        gat_num_heads=args.gat_num_heads
    )


def load_embeddings(filepath):
    """
    Load and process embedding data
    """
    try:
        uniprot2embed = pd.read_csv(filepath, sep = '\t', compression = 'gzip', index_col = 0)
        
        # Set index and clean data
        uniprot2embed.index = uniprot2embed['uniprot_id']
        
        # Check for required columns
        if 'sequence' in uniprot2embed.columns:
            uniprot2embed = uniprot2embed.drop(columns=['uniprot_id', 'sequence'])
        else:
            uniprot2embed = uniprot2embed.drop(columns=['uniprot_id'])
        
        uniprot2embed = uniprot2embed.drop_duplicates()
        
        # Convert to dictionary format
        uniprot2embed_dict = {idx: np.array(row) for idx, row in uniprot2embed.iterrows()}
        target_to_index = {value: index for index, value in enumerate(uniprot2embed.index)}
        
        print(f"Loaded embeddings for {len(uniprot2embed_dict)} proteins")
        print(f"Embedding dimension: {uniprot2embed.shape[1]}")
        
        return uniprot2embed_dict, uniprot2embed, target_to_index
        
    except Exception as e:
        raise ValueError(f"Error loading embeddings: {e}")


def predict_expression(model, embeddings_tensor, batch_size, device):
    """
    Predict gene expression from embeddings
    """
    try:
        dataset = TensorDataset(embeddings_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_results = []
        model.eval()
        
        print("Starting prediction...")
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing batches"):
                try:
                    batch_data = batch[0].to(device, non_blocking=True)
                    batch_results = model(batch_data)
                    all_results.append(batch_results.cpu())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\nGPU out of memory error occurred!")
                        print(f"Current batch size: {batch_size}")
                        print("Try reducing the batch size with --batch_size parameter")
                        raise e
                    else:
                        raise e
        
        # Combine results
        final_result = torch.cat(all_results, dim=0)
        print(f"Prediction completed. Output shape: {final_result.shape}")
        
        return final_result
        
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")


def save_results(results_tensor, gene_list, uniprot_ids, output_dir, filename):
    """
    Save prediction results to file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame
        final_result_df = pd.DataFrame(results_tensor.numpy())
        final_result_df.columns = gene_list
        final_result_df['UniProt'] = list(uniprot_ids)
        
        # Save to file
        output_path = os.path.join(output_dir, filename)
        final_result_df.to_csv(output_path, index=False)
        
        print(f"Results saved to {output_path}")
        print(f"Output shape: {final_result_df.shape}")
        
    except Exception as e:
        raise IOError(f"Error saving results: {e}")


def main():
    """
    Main function to run TE prediction from embeddings
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate Target-Gene Expression predictions from protein sequence embeddings"
    )
    
    parser.add_argument(
        '--input_embed_file', 
        required=True,
        help="Path to the input embedding file containing UniProt IDs and embeddings"
    )
    
    parser.add_argument(
        '--save_dir', 
        required=True,
        help="Directory to save the output TE matrix file"
    )
    
    parser.add_argument(
        '--prefix', 
        required=True,
        help="Prefix for the output file (recommend to match with the input file name)"
    )
    
    parser.add_argument(
        '--cell', 
        required=True,
        help='Cell line for TDEM (e.g., HepG2, MCF7, etc.)'
    )
    
    parser.add_argument(
        '--gene_space', 
        default='best_inferred', 
        choices=['landmark', 'best_inferred'],
        help="Gene space to use: 'landmark' or 'best_inferred'"
    )
    
    parser.add_argument(
        '--best_model_path', 
        required=True,
        help="Path to the best model file (.pth)"
    )
    
    parser.add_argument(
        '--device_num', 
        default=0, 
        type=int,
        help="GPU device ID (default: 0)"
    )
    
    parser.add_argument(
        '--batch_size', 
        default=1000, 
        type=int,
        help="Batch size for generating embeddings (default: 1000)"
    )
    
    parser.add_argument(
        '--hidden_dims', 
        nargs='+', 
        type=int, 
        default=[512, 256, 512],
        help='Hidden dimensions for the model layers (default: [512, 256, 512])'
    )
    
    parser.add_argument(
        '--graph_node_embed_dim', 
        type=int, 
        default=32,
        help='Node embedding dimension in graph learning (default: 32)'
    )
    
    parser.add_argument(
        '--gat_num_heads', 
        type=int, 
        default=3,
        help='Number of heads in GAT (default: 3)'
    )
    
    parser.add_argument(
        '--datadir',
        default='./data',
        help='Directory containing data files (default: ./data)'
    )
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device(f'cuda:{args.device_num}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    
    print('Starting Target-Gene Expression prediction')
    print(f'Input embedding file: {args.input_embed_file}')
    print(f'Output directory: {args.save_dir}')
    print(f'Cell line: {args.cell}')
    print(f'Using device: {device}')
    
    try:
        # Validate input files
        validate_input_file(args.input_embed_file)
        validate_model_file(args.best_model_path)
        
        # Load embeddings
        print("Loading embeddings...")
        uniprot2embed_dict, uniprot2embed_df, target_to_index = load_embeddings(args.input_embed_file)
        args.target_embed_dim = uniprot2embed_df.shape[1]
        
        # Load gene information
        print("Loading gene information...")
        gene_list, cmap_ginfo = load_gene_info(args.datadir, args.gene_space)
        args.gene_dim = len(gene_list)
        args.genel = gene_list
        
        # Load STRING data
        print("Loading PPI data...")
        string_df_f, ent2id = load_string_data(args.datadir, gene_list)
        args.ppi_edge_data = string_df_f
        
        # Create graph data
        graph_data = create_graph_data_without_score(string_df_f)
        args.graph_data = graph_data
        
        # Initialize model
        print("Initializing model...")
        target2exp = generate_TE(args).to(device)
        
        # Load model weights
        print("Loading model weights...")
        try:
            state_dict = torch.load(args.best_model_path, map_location=device)
            target2exp.load_state_dict(state_dict)
            print("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Prepare data for prediction
        print("Preparing data for prediction...")
        all_embeddings = list(uniprot2embed_dict.values())
        embeddings_tensor = torch.tensor(all_embeddings, dtype=torch.float32)
        
        # Make predictions
        results_tensor = predict_expression(
            target2exp, 
            embeddings_tensor, 
            args.batch_size, 
            device
        )
        
        # Save results
        output_filename = f"{args.prefix}_{args.cell}_TE_matrix.csv"
        save_results(
            results_tensor,
            gene_list,
            uniprot2embed_dict.keys(),
            args.save_dir,
            output_filename
        )
        
        print("TE prediction completed successfully!")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
