import os, sys, pickle, importlib, time, random, multiprocessing, requests, argparse, datetime, ast, re
from tqdm import tqdm
from itertools import chain, repeat
import scipy.stats
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score, f1_score, recall_score, precision_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

from Bio.PDB import PDBList
from Bio import SeqIO
from transformers import BertModel, BertTokenizer




def load_de_matrix(de_path, check_qc_pass=True):
    """
    Load the Drug-induced gene expression (DE) matrix from a file.
    Inputs:
        de_path (str): Path to the DE file (.pkl or .csv).
        check_qc_pass (bool): Whether to filter rows based on the 'qc_pass' column if it exists.
    Returns:
        pd.DataFrame of filtered DE matrix.
    Raises:
        ValueError: If the file extension is unsupported or the required 'sig_id', 'pert_id','pubchem_cid' column is missing.
    """
    # Load DE matrix
    if de_path.endswith('.pkl'):
        de = pd.read_pickle(de_path)
    elif de_path.endswith('.csv'):
        de = pd.read_csv(de_path)
    else:
        raise ValueError("DE dataframe file extension is not supported, it should be either .pkl or .csv")
    
    # Check if 'pubchem_cid' column exists
    if not all(col in de.columns for col in ['sig_id', 'pert_id', 'pubchem_cid']):
        raise ValueError("DE dataframe does not have 'sig_id', 'pert_id',or 'pubchem_cid' column, which are required for DE dataframe")
    
    # Ensure 'pubchem_cid' is of integer type
    de = de[de['pubchem_cid'].notna()]
    de['pubchem_cid'] = de['pubchem_cid'].astype(int)

    # Optional: Filter based on 'qc_pass' column
    if check_qc_pass:
        if 'qc_pass' in de.columns:
            de = de[de['qc_pass'] == True]
        else:
            print("Warning: 'qc_pass' column does not exist in DE dataframe, skipping qc_pass filtering")

    return de


def load_dti_matrix(dti_path):
    """
    Load the Drug-Target Interaction (DTI) matrix from a file.
    Inputs:
        dti_path (str): Path to the DTI file (.tsv, .csv, or .pkl).
    Returns:
        pd.DataFrame of the loaded DTI matrix.
    Raises:
        ValueError: If the file extension is unsupported or the required 'CID' and 'UniProt' columns are missing.
    """
    # Load DTI matrix
    if dti_path.endswith('.tsv'):
        dti = pd.read_csv(dti_path, sep='\t')
    elif dti_path.endswith('.csv'):
        dti = pd.read_csv(dti_path, sep=',')
    elif dti_path.endswith('.pkl'):
        dti = pd.read_pickle(dti_path)
    else:
        raise ValueError("DTI dataframe file extension is not supported, it should be either .tsv, .csv, or .pkl")

    # Check if 'CID' and 'UniProt' columns exist
    if not {'CID', 'UniProt'}.issubset(dti.columns):
        raise ValueError("DTI dataframe does not have 'CID' and 'UniProt' columns, which are required for DTI dataframe")

    # Ensure 'CID' is of integer type
    dti['CID'] = dti['CID'].astype(int)

    return dti


def load_protbert_embedding(protbert_embed_path):
    """
    Load the ProtBERT embedding file.

    Args:
        protbert_embed_path (str): Path to the ProtBERT embedding file (.tsv, .csv, .gz, or .pkl).

    Returns:
        pd.DataFrame: The loaded ProtBERT embedding dataframe.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    # Determine file type and load accordingly
    if protbert_embed_path.endswith('.tsv'):
        uniprot2bert = pd.read_csv(protbert_embed_path, sep='\t')
    elif protbert_embed_path.endswith('.csv'):
        uniprot2bert = pd.read_csv(protbert_embed_path)
    elif protbert_embed_path.endswith('.gz'):
        uniprot2bert = pd.read_csv(protbert_embed_path, sep='\t', compression='gzip')
    elif protbert_embed_path.endswith('.pkl'):
        uniprot2bert = pd.read_pickle(protbert_embed_path)
    else:
        raise ValueError("ProtBERT embedding file extension is not supported. Supported extensions: .tsv, .csv, .gz, .pkl")
    
    uniprot2bert.set_index('uniprot_id', inplace=True)
    return uniprot2bert



def get_amino_acid_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        # Extract the sequence from the FASTA format
        sequence = "".join(fasta_data.split("\n")[1:])
        return sequence
    else:
        print(f"Failed to retrieve data for UniProt ID {uniprot_id}")
        return None


def fetch_uniprot_sequences(protein_names, max_results = 20):
    if isinstance(protein_names, str):
        protein_names = [protein_names]
    sequences = {}
    for protein_name in tqdm(protein_names):
        base_url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": protein_name,  # 단백질 이름 검색
            "format": "fasta",      # FASTA 형식으로 가져오기
            "size": max_results     # 최대 가져올 개수
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            fasta_data = response.text
            current_id = None
            for line in fasta_data.split("\n"):
                if line.startswith(">"):
                    current_id = line.split("|")[1]  # UniProt ID 가져오기
                    sequences[current_id] = ""
                elif current_id:
                    sequences[current_id] += line.strip()  # 서열 추가
        else:
            print(f"Error {response.status_code}: '{protein_name}' 데이터 검색 실패")
    return sequences


def get_protein_embeddings_in_batches(sequences, device_id, model_name="Rostlab/prot_bert_bfd", batch_size=5):
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name)
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    embeddings = []
    # Process sequences in batches
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i + batch_size]
        inputs = tokenizer(batch_sequences, return_tensors="pt", max_length=1024, truncation=True, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        #
        with torch.no_grad():
            outputs = model(**inputs)
        #
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        #del inputs
        #del outputs
        #torch.cuda.empty_cache()
        embeddings.extend(batch_embeddings)
    return embeddings


def filter_DTI_until_convergence(DTI, min_cut = 3, max_cut = 110):
    prev_shape = None  
    current_shape = DTI.shape
    iteration = 0
    while prev_shape != current_shape:
        iteration += 1
        prev_shape = current_shape
        DTI = DTI.loc[:, DTI.sum() >= min_cut]
        DTI = DTI.loc[DTI.sum(axis=1) <= max_cut]
        DTI = DTI[~(DTI == 0).all(axis=1)]
        DTI = DTI.T[~(DTI.T == 0).all(axis=1)].T
        current_shape = DTI.shape
        print(f"Iteration {iteration}: New shape = {current_shape}")
    print(f'Final number of drug : {DTI.shape[0]}, target : {DTI.shape[1]}')
    return DTI


def match_space_CMAP(dt, de, min_cut=3, max_cut=110):
    compl = set(dt.index)
    compl &= set(de['pubchem_cid'])
    genel = set(de.columns[de.columns.str.match(r'^\d+$')])
    compl = list(compl)
    genel = list(genel)
    de = de[de['pubchem_cid'].isin(compl)].loc[:, genel + ['sig_id', 'pert_id', 'pubchem_cid']]
    dt = dt.loc[compl]
    dt = filter_DTI_until_convergence(dt, min_cut=min_cut, max_cut=max_cut)
    return dt, de

def save_model_parameters(hyperparam_file, args, parser):
    with open(hyperparam_file, 'w') as f:
        f.write('========================================\n')
        f.write('Model Parameters:\n')
        f.write('========================================\n')
        # Loop over all attributes in args, not only those in parser
        for arg_name, arg_value in vars(args).items():
            f.write(f'{arg_name}: {arg_value}\n')


def load_ppi_data(ppi_data_path, genel):
    """
    Process Protein-Protein Interaction (PPI) data and create graph data.

    Args:
        args (Namespace): Contains gene list (`genel`) and will store processed data in `args.ppi_edge_data` and `args.graph_data`.
    Returns:
        pd.DataFrame: Filtered PPI edge data.
        Any: Graph data created using `create_graph_data_without_score`.
    """
    # Map genes to IDs
    ent2id = {gene: idx for idx, gene in enumerate(genel)}

    # Load and map PPI data
    string_df = pd.read_csv(ppi_data_path)
    string_df['ens1'] = string_df.ent1.astype(str).map(ent2id)
    string_df['ens2'] = string_df.ent2.astype(str).map(ent2id)

    # Filter mapped data
    string_df_f = string_df.loc[:, ['ens1', 'ens2']].dropna()

    # Create graph data
    graph_data = create_graph_data_without_score(string_df_f)

    return graph_data

#------------------------------------------------------------

class DTI_dataset_with_sampled_true_neg(Dataset):
    def __init__(self, args, dataframe):
        self.dataframe = dataframe
        self.positive_data = list(zip(
            dataframe[dataframe['Binding'] == 1]['DE'],
            dataframe[dataframe['Binding'] == 1]['UniProt_embed'],
            dataframe[dataframe['Binding'] == 1]['CID'],
            dataframe[dataframe['Binding'] == 1]['UniProt']
        ))
        self.negative_data = list(zip(
            dataframe[dataframe['Binding'] == 0]['DE'],
            dataframe[dataframe['Binding'] == 0]['UniProt_embed'],
            dataframe[dataframe['Binding'] == 0]['CID'],
            dataframe[dataframe['Binding'] == 0]['UniProt']
        ))
        self.pos_neg_ratio = args.pos_neg_ratio

    def __len__(self):
        return len(self.positive_data)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")
        
        # Get positive sample
        cid, uniprot, cid_id, uniprot_id = self.positive_data[idx]
        drug_exp = torch.tensor(cid, dtype=torch.float32).clone().detach()
        uniprot_embed = torch.tensor(uniprot.astype(np.float32), dtype=torch.float32).clone().detach()
        label = torch.tensor(1, dtype=torch.float32)  # Positive sample label
        
        # Generate negative samples by random sampling
        negative_samples = []
        for _ in range(self.pos_neg_ratio):
            negative_cid, negative_uniprot, negative_cid_id, negative_uniprot_id = random.choice(self.negative_data)
            negative_cid_exp = torch.tensor(negative_cid, dtype=torch.float32).clone().detach()
            negative_uniprot_embed = torch.tensor(negative_uniprot.astype(np.float32), dtype=torch.float32).clone().detach()
            negative_label = torch.tensor(0, dtype=torch.float32)  # Negative sample label
            
            # Store negative sample in dictionary with CID and UniProt ID
            negative_samples.append({
                'DE': negative_cid_exp,
                'UniProt_embed': negative_uniprot_embed,
                'label': negative_label,
                'CID': negative_cid_id,
                'UniProt': negative_uniprot_id
            })
        
        # Return the positive and negative samples as a dictionary with CID and UniProt ID
        sample = {
            'positive': {
                'DE': drug_exp,
                'UniProt_embed': uniprot_embed,
                'label': label,
                'CID': cid_id,
                'UniProt': uniprot_id
            },
            'negative': negative_samples
        }
        return sample


class DTI_dataset_for_prediction(Dataset):
    def __init__(self, args, dataframe):
        """
        Dataset for DTI data that directly uses labels without generating negative samples.

        Args:
            args (Namespace): Contains configurations (e.g., batch size).
            dataframe (pd.DataFrame): The input dataframe with DE, UniProt_embed, CID, UniProt columns.
        """
        self.dataframe = dataframe
        self.data = list(zip(
            dataframe['DE'],
            dataframe['UniProt_embed'],
            dataframe['CID'],
            dataframe['UniProt'],
            dataframe['Binding']
        ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")

        # Extract sample
        cid, uniprot, cid_id, uniprot_id, label = self.data[idx]

        # Convert to tensors
        drug_exp = torch.tensor(cid, dtype=torch.float32).clone().detach()
        uniprot_embed = torch.tensor(uniprot.astype(np.float32), dtype=torch.float32).clone().detach()

        # Return sample
        return {
            'DE': drug_exp,
            'UniProt_embed': uniprot_embed,
            'CID': cid_id,
            'UniProt': uniprot_id,
            'label': torch.tensor(label, dtype=torch.float32)  # Use the actual label from the dataframe
        }    

class generate_TE(nn.Module):
    def __init__(self, args):
        super(generate_TE, self).__init__()
        self.input_dim = args.target_embed_dim
        self.hidden_dims = args.hidden_dims
        self.output_dim = args.gene_dim
        self.graph_node_embed_dim = args.graph_node_embed_dim
        self.num_heads = args.gat_num_heads
        self.graph_data = args.graph_data
        self.dropout = args.dropout
        self.ablation_mode = args.ablation_mode
        self.fc_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            self.fc_layers.append(nn.Linear(self.input_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.input_dim = hidden_dim
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)        
        # without pretrained embedding (no_pretrained ablation)
        if self.ablation_mode == 'no_pretrained':
            self.protein_embedding = nn.Embedding(args.num_proteins, args.target_embed_dim)
            nn.init.normal_(self.protein_embedding.weight, mean=0, std=0.1)
        if self.ablation_mode != 'no_gat':
            self.embedding = nn.Linear(1, self.graph_node_embed_dim)
            self.gat = GATConv(
                in_channels=self.graph_node_embed_dim,
                out_channels=self.graph_node_embed_dim,
                heads=self.num_heads,
                concat=False,
                dropout=self.dropout
            )
            self.gat_output = nn.Linear(self.graph_node_embed_dim, 1)
            self.register_buffer('edge_index', self.graph_data.edge_index)
            self.register_buffer('edge_weight', self.graph_data.edge_attr)
    #
    def forward(self, x, protein_ids=None):
        batch_size = x.size(0)
        if self.ablation_mode == 'no_pretrained' and protein_ids is not None:
            x = self.protein_embedding(protein_ids)
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)        
        x = self.output_layer(x)
        # no_gat ablation: 직접 출력 반환
        if self.ablation_mode == 'no_gat':
            return x        
        x_emb = x.unsqueeze(-1)
        x_emb = self.embedding(x_emb)        
        gat_outputs = []
        for i in range(batch_size):
            node_features = x_emb[i]
            gat_out = self.gat(node_features, self.edge_index, self.edge_weight)
            gat_out = self.gat_output(gat_out)
            gat_out = gat_out.squeeze(-1)
            gat_outputs.append(gat_out)
        output = torch.stack(gat_outputs, dim=0)
        return output

class SimilarityScore(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, ablation_mode='full'):
        super().__init__()
        self.ablation_mode = ablation_mode
        if ablation_mode == 'no_correlation':
            self.alpha = 0.0  
            self.beta = 1.0  
        elif ablation_mode == 'no_dot_product':
            self.alpha = 1.0  
            self.beta = 0.0   
        else:
            self.alpha = alpha
            self.beta = beta    
    def pearson_correlation(self, pred, target):
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)
        covariance = (pred_centered * target_centered).mean(dim=1)
        pred_std = pred_centered.std(dim=1)
        target_std = target_centered.std(dim=1)
        correlation = covariance / (pred_std * target_std + 1e-8)
        return correlation
    def forward(self, output, target):
        abs_target = torch.abs(target)
        weights = abs_target / (abs_target.sum(dim=1, keepdim=True) + 1e-8)
        dot_product_score = torch.sum(output * target * weights, dim=1)
        pearson_score = self.pearson_correlation(output, target)        
        total_score = self.alpha * dot_product_score + self.beta * pearson_score
        return total_score



def calculate_metrics(group):
    # Extract predictions and labels
    labels = group['label'].values
    prediction = group['prediction'].values
    # Sort by prediction in descending order for top-N accuracy
    sorted_group = group.sort_values(by='prediction', ascending=False)
    top_100_correct = sorted_group.head(100)['label'].sum()
    top_30_correct = sorted_group.head(30)['label'].sum()
    top_20_correct = sorted_group.head(20)['label'].sum()
    # Basic metrics
    metrics = {}
    metrics['AUROC'] = roc_auc_score(labels, prediction) if len(np.unique(labels)) > 1 else np.nan
    metrics['AUPRC'] = average_precision_score(labels, prediction) if len(np.unique(labels)) > 1 else np.nan
    # Precision, Recall, F1-Score
    predicted = prediction >= 0.5  # Threshold = 0.5 for classification
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary')
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1_Score'] = f1
    # Accuracy
    metrics['Accuracy'] = accuracy_score(labels, predicted)
    # Top-N Accuracies
    metrics['Top_100_Accuracy'] = top_100_correct / 100 if len(sorted_group) >= 100 else np.nan
    metrics['Top_30_Accuracy'] = top_30_correct / 30 if len(sorted_group) >= 30 else np.nan
    metrics['Top_20_Accuracy'] = top_20_correct / 20 if len(sorted_group) >= 20 else np.nan
    return pd.Series(metrics)


def calculate_metrics_during_training(labels, probabilities):
    probabilities = np.array(probabilities)  # Convert list to NumPy array
    sorted_indices = np.argsort(-probabilities)  # Descending sort
    sorted_labels = np.array(labels)[sorted_indices]
    top_100_correct = sorted_labels[:100].sum() if len(sorted_labels) >= 100 else np.nan
    top_30_correct = sorted_labels[:30].sum() if len(sorted_labels) >= 30 else np.nan

    metrics = {}
    metrics['AUROC'] = roc_auc_score(labels, probabilities) if len(np.unique(labels)) > 1 else np.nan
    metrics['AUPRC'] = average_precision_score(labels, probabilities) if len(np.unique(labels)) > 1 else np.nan

    # Classification metrics
    predicted = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary')
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1_Score'] = f1
    metrics['Accuracy'] = accuracy_score(labels, predicted)

    # Top-N Accuracies
    metrics['Top_100_Accuracy'] = top_100_correct / 100 if not np.isnan(top_100_correct) else np.nan
    metrics['Top_30_Accuracy'] = top_30_correct / 30 if not np.isnan(top_30_correct) else np.nan
    return metrics



