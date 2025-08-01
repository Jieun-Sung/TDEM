'''
Embed protein sequences using Pre-trained Language Models (PLM)

Usage:
python embed_sequence_with_PLM.py \
    --input_seqfile /path/to/uniprot2sequence.tsv \
    --PLM esm2 \
    --prefix test \
    --output_dir /path/to/output_directory \
    --batch_size 8

Then, the embeddings will be saved in the specified output directory with the name test_esm2_embeddings.tsv.gz.
'''

import torch
import os
import sys
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse


def validate_input_file(filepath):
    """
    Validate input file exists and has proper format
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    try:
        # Check if file can be read and has proper format
        df = pd.read_csv(filepath, sep='\t', header=None, names=['uniprot_id', 'sequence'], nrows=5)
        if len(df.columns) != 2:
            raise ValueError("Input file must have exactly 2 columns: uniprot_id and sequence")
        
        # Check for empty sequences
        if df['sequence'].isnull().any() or (df['sequence'] == '').any():
            print("Warning: Found empty sequences in the input file. They will be filtered out.")
            
    except Exception as e:
        raise ValueError(f"Error reading input file: {e}")


def load_model_and_tokenizer(plm_name):
    """
    Load the specified PLM model and tokenizer
    """
    print(f'Loading PLM model: {plm_name}...')
    
    try:
        if plm_name == 'esm2':
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        elif plm_name == 'protbert':
            tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
            model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
        else:
            raise ValueError(f"PLM '{plm_name}' is not implemented yet. Available options: esm2, protbert")
        
        print(f'PLM {plm_name} loaded successfully.')
        return tokenizer, model
        
    except Exception as e:
        print(f"Error loading model {plm_name}: {e}")
        print("This might be due to:")
        print("1. Network connection issues")
        print("2. Insufficient disk space")
        print("3. Model not available")
        sys.exit(1)


def create_collate_fn(tokenizer):
    """
    Create collate function with proper tokenizer scope
    """
    def collate_fn(batch):
        return tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1022)
    
    return collate_fn


class SeqDataset(Dataset):
    """
    Dataset class for protein sequences
    """
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def compute_embeddings(model, tokenizer, sequences, batch_size, device):
    """
    Compute embeddings for protein sequences
    """
    dataset = SeqDataset(sequences)
    collate_fn = create_collate_fn(tokenizer)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=0, 
        pin_memory=True
    )
    
    print(f"DataLoader created with batch size {batch_size} and {len(dataset)} sequences.")
    print("Starting to compute embeddings...")
    
    emb_list = []
    model.eval()
    
    try:
        with torch.no_grad():
            for inputs in tqdm(loader, desc="Processing batches"):
                try:
                    # Move inputs to device
                    for k in inputs:
                        inputs[k] = inputs[k].to(device, non_blocking=True)
                    
                    # Get model outputs
                    outputs = model(**inputs)
                    emb = outputs['last_hidden_state'].detach()
                    
                    # Mean pooling across sequence length
                    emb = emb.mean(dim=1)
                    emb_list.append(emb.cpu())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\nGPU out of memory error occurred!")
                        print(f"Current batch size: {batch_size}")
                        print("Try reducing the batch size with --batch_size parameter")
                        print("Or use CPU by setting CUDA_VISIBLE_DEVICES=''")
                        raise e
                    else:
                        raise e
                        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        sys.exit(1)
    
    # Concatenate all embeddings
    all_emb = torch.cat(emb_list, dim=0)
    return all_emb


def save_embeddings(embeddings, uniprot_ids, output_path):
    """
    Save embeddings to file
    """
    # Create DataFrame with embeddings
    emb_df = pd.DataFrame(embeddings.numpy(), index=uniprot_ids)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to compressed TSV file
    emb_df.to_csv(output_path, sep='\t', compression='gzip')
    print(f"Embeddings saved to {output_path}")
    print(f"Final embedding shape: {emb_df.shape}")


def main():
    """
    Main function to run the embedding computation
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Embed protein sequences using Pre-trained Language Models (PLM).'
    )
    
    parser.add_argument(
        '--input_seqfile', 
        type=str, 
        required=True,
        help='Path to the input sequence file containing sequences in two columns: uniprot_id and sequence. '
             'Should be tab-separated two columns data with uniprot_id in the first column and sequence in the second column.'
    )
    
    parser.add_argument(
        '--PLM', 
        type=str, 
        default='esm2', 
        choices=['esm2', 'protbert'],
        help='Pre-trained language model to use for embedding sequences. '
             'Options: esm2, protbert. Note that esm2 is the default and recommended model for protein sequences.'
    )
    
    parser.add_argument(
        '--prefix', 
        type=str, 
        default='test',
        help='Prefix for the output files. Default is "test".'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='.',
        help='Directory to save the output embeddings. Default is current directory.'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=8,
        help='Batch size for processing sequences. Default is 8. '
             'Reduce this value if you encounter GPU memory issues.'
    )
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Start embedding computation with PLM {args.PLM} with {args.input_seqfile}')
    print('Using device:', device)
    
    try:
        # Validate input file
        validate_input_file(args.input_seqfile)
        
        # Load model and tokenizer
        tokenizer, model = load_model_and_tokenizer(args.PLM)
        
        # Move model to device
        model = model.to(device)
        
        # Load and preprocess data
        print(f"Loading sequences from {args.input_seqfile}...")
        uniprot2seq = pd.read_csv(args.input_seqfile, sep='\t', header=None, names=['uniprot_id', 'sequence'])
        
        # Filter out empty sequences
        initial_count = len(uniprot2seq)
        uniprot2seq = uniprot2seq.dropna(subset=['sequence'])
        uniprot2seq = uniprot2seq[uniprot2seq['sequence'] != '']
        final_count = len(uniprot2seq)
        
        if initial_count > final_count:
            print(f"Filtered out {initial_count - final_count} empty sequences.")
        
        if final_count == 0:
            print("Error: No valid sequences found in the input file.")
            sys.exit(1)
        
        print(f"Loaded {final_count} valid sequences from {args.input_seqfile}")
        
        sequences = uniprot2seq['sequence'].tolist()
        uniprot_ids = uniprot2seq['uniprot_id'].tolist()
        
        # Compute embeddings
        embeddings = compute_embeddings(model, tokenizer, sequences, args.batch_size, device)
        
        # Save results
        output_path = os.path.join(args.output_dir, f"{args.prefix}_{args.PLM}_embeddings.tsv.gz")
        save_embeddings(embeddings, uniprot_ids, output_path)
        
        print("Embedding computation completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
