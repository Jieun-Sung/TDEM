'''
Extract protein sequences from PDB files using UniProt IDs

Usage:
python get_uniprot_sequences.py \
    --input_seqfile /path/to/uniprot_id.csv \
    --prefix test \
    --output_dir /path/to/output_directory \
    --ncore 40

input file uniprot_id.csv should contain a column named 'uniprot_id' or 'UniProt'.
Then, the sequences will be saved as test_uniprot2sequence.tsv in the specified output directory.
'''

import pandas as pd
import os
import sys
import re
import multiprocessing
from Bio import SeqIO
from Bio.PDB import PDBList
from tqdm import tqdm
import argparse


def validate_input_file(filepath):
    """
    Validate input file exists and has proper format
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    try:
        # Check if file can be read
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, nrows=5)
        else:
            df = pd.read_csv(filepath, sep='\t', nrows=5)
        
        if df.empty:
            raise ValueError("Input file is empty")
            
        print(f"Input file validation successful. Found {len(df)} sample rows.")
        
    except Exception as e:
        raise ValueError(f"Error reading input file: {e}")


def get_protein_sequence_from_pdb(pdb_file_path):
    """
    Extracts the protein sequence from a downloaded PDB file.
    """
    try:
        sequences = []
        for record in SeqIO.parse(pdb_file_path, "pdb-seqres"):
            sequences.append(str(record.seq))
        
        if sequences:
            # Return the first sequence found in the PDB file
            return sequences[0]
        else:
            return None
    except Exception as e:
        print(f"Error processing {pdb_file_path}: {e}")
        return None


def get_pdb_id(pdb_file_name):
    """
    Extract PDB ID from PDB file name
    """
    match = re.match(r'pdb(\w{4})\.ent', pdb_file_name)
    if match:
        return match.group(1)
    return None


def download_pdb_files(pdb_ids, pdb_dir, ncore=8):
    """
    Download PDB files if they don't exist
    """
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    
    existing_files = set(os.listdir(pdb_dir))
    pdb_ids_to_download = []
    
    for pdb_id in pdb_ids:
        expected_filename = f"pdb{pdb_id.lower()}.ent"
        if expected_filename not in existing_files:
            pdb_ids_to_download.append(pdb_id)
    
    if pdb_ids_to_download:
        print(f"Downloading {len(pdb_ids_to_download)} PDB files...")
        pdbl = PDBList()
        
        for pdb_id in tqdm(pdb_ids_to_download, desc="Downloading PDB files"):
            try:
                pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=pdb_dir, overwrite=False)
            except Exception as e:
                print(f"Failed to download {pdb_id}: {e}")
    else:
        print("All PDB files already exist.")


def process_sequences_parallel(pdb_files, ncore):
    """
    Process PDB files in parallel to extract sequences
    """
    print(f"Processing {len(pdb_files)} PDB files using {ncore} cores...")
    
    try:
        with multiprocessing.Pool(processes=ncore) as pool:
            ptn_sequences = pool.map(get_protein_sequence_from_pdb, pdb_files)
        return ptn_sequences
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        print("Falling back to sequential processing...")
        return [get_protein_sequence_from_pdb(pdb_file) for pdb_file in tqdm(pdb_files, desc="Processing PDB files")]


def main():
    """
    Main function to extract protein sequences from PDB files
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Extract protein sequences from PDB files using UniProt IDs.'
    )
    
    parser.add_argument(
        '--input_seqfile', 
        type=str, 
        required=True,
        help='Path to the input CSV file containing UniProt IDs. '
             'Should contain uniprot_id column or be a single column with UniProt IDs.'
    )
    
    parser.add_argument(
        '--uniprot2pdb_file',
        type=str,
        default='/spstorage/USERS/sung/DATA/Uniprot_PDB_files/uniprot2pdb.tsv',
        help='Path to the uniprot2pdb mapping file. Default: /spstorage/USERS/sung/DATA/Uniprot_PDB_files/uniprot2pdb.tsv'
    )
    
    parser.add_argument(
        '--pdb_dir',
        type=str,
        default='./pdb_files',
        help='Directory to store downloaded PDB files. Default: ./pdb_files'
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
        help='Directory to save the output file. Default is current directory.'
    )
    
    parser.add_argument(
        '--ncore', 
        type=int, 
        default=8,
        help='Number of CPU cores to use for parallel processing. Default is 8.'
    )
    
    parser.add_argument(
        '--download_pdb',
        action='store_true',
        help='Download missing PDB files automatically. Default is False.'
    )
    
    args = parser.parse_args()
    
    print('Starting protein sequence extraction from PDB files')
    print(f'Input file: {args.input_seqfile}')
    print(f'Output directory: {args.output_dir}')
    print(f'Using {args.ncore} CPU cores')
    
    try:
        # Validate input file
        validate_input_file(args.input_seqfile)
        
        # Load uniprot IDs
        print(f"Loading UniProt IDs from {args.input_seqfile}...")
        if args.input_seqfile.endswith('.csv'):
            uniprot_df = pd.read_csv(args.input_seqfile)
        else:
            uniprot_df = pd.read_csv(args.input_seqfile, sep='\t')
        
        # Try to find uniprot_id column or use first column
        if 'uniprot_id' in uniprot_df.columns:
            uniprot_ids = uniprot_df['uniprot_id'].unique().tolist()
        elif 'UniProt' in uniprot_df.columns:
            uniprot_ids = uniprot_df['UniProt'].unique().tolist()
        else:
            # Use first column
            uniprot_ids = uniprot_df.iloc[:, 0].unique().tolist()
        
        print(f"Found {len(uniprot_ids)} unique UniProt IDs")
        
        # Load uniprot2pdb mapping
        print(f"Loading UniProt to PDB mapping from {args.uniprot2pdb_file}...")
        if not os.path.exists(args.uniprot2pdb_file):
            raise FileNotFoundError(f"UniProt to PDB mapping file not found: {args.uniprot2pdb_file}")
        
        uniprot2pdb = pd.read_csv(args.uniprot2pdb_file, sep='\t')
        print(f"Loaded {len(uniprot2pdb)} UniProt-PDB mappings")
        
        # Filter for requested UniProt IDs
        filtered_mapping = uniprot2pdb[uniprot2pdb['From'].isin(uniprot_ids)]
        print(f"Found PDB mappings for {len(filtered_mapping)} entries")
        
        if len(filtered_mapping) == 0:
            print("Warning: No PDB mappings found for the provided UniProt IDs")
            return
        
        # Get unique PDB IDs
        pdb_ids = filtered_mapping['To'].unique().tolist()
        print(f"Need to process {len(pdb_ids)} unique PDB files")
        
        # Download PDB files if requested
        if args.download_pdb:
            download_pdb_files(pdb_ids, args.pdb_dir, args.ncore)
        
        # Get list of available PDB files
        if not os.path.exists(args.pdb_dir):
            raise FileNotFoundError(f"PDB directory not found: {args.pdb_dir}. Use --download_pdb to download files automatically.")
        
        available_pdb_files = os.listdir(args.pdb_dir)
        pdb_files = [os.path.join(args.pdb_dir, file) for file in available_pdb_files if file.endswith('.ent')]
        
        if not pdb_files:
            raise FileNotFoundError(f"No PDB files found in {args.pdb_dir}")
        
        print(f"Found {len(pdb_files)} PDB files to process")
        
        # Extract sequences in parallel
        ptn_sequences = process_sequences_parallel(pdb_files, args.ncore)
        
        # Create DataFrame with results
        pdb2sequence = pd.DataFrame({
            'pdb_file': [os.path.basename(f) for f in pdb_files], 
            'sequence': ptn_sequences
        })
        
        # Extract PDB IDs from filenames
        pdb2sequence['pdb_id'] = [get_pdb_id(file) for file in pdb2sequence['pdb_file']]
        pdb2sequence['pdb_id'] = pdb2sequence['pdb_id'].str.upper()
        
        # Filter out entries with no sequences
        valid_sequences = pdb2sequence.dropna(subset=['sequence'])
        print(f"Successfully extracted sequences from {len(valid_sequences)}/{len(pdb2sequence)} PDB files")
        
        # Merge with UniProt mapping
        uniprot2sequence = pd.merge(
            filtered_mapping, 
            valid_sequences, 
            left_on='To', 
            right_on='pdb_id', 
            how='left'
        )
        
        # Select and rename columns
        final_result = uniprot2sequence[['From', 'sequence']].copy()
        final_result.columns = ['uniprot_id', 'sequence']
        
        # Remove duplicates and entries without sequences
        final_result = final_result.dropna(subset=['sequence']).drop_duplicates(subset=['uniprot_id'])
        
        print(f"Final result: {len(final_result)} UniProt IDs with sequences")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{args.prefix}_uniprot2sequence.tsv")
        
        final_result.to_csv(output_path, sep='\t', index=False)
        print(f"Results saved to {output_path}")
        
        # Print summary statistics
        print("\n=== Summary ===")
        print(f"Input UniProt IDs: {len(uniprot_ids)}")
        print(f"UniProt IDs with PDB mappings: {len(filtered_mapping['From'].unique())}")
        print(f"Unique PDB IDs: {len(pdb_ids)}")
        print(f"Successfully processed PDB files: {len(valid_sequences)}")
        print(f"Final UniProt IDs with sequences: {len(final_result)}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
