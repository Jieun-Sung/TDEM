
import os,sys

from utils import *

###========================================================
### ARGUMENTS 
###========================================================

parser = argparse.ArgumentParser(description='Learning TDEM')
parser.add_argument('--cell', type=str, required=True, 
        choices=['A375', 'A549', 'MCF7', 'PC3', 'HT29', 'HA1E'],
        help='Cell line to use for training and evaluation. Choose from: A375, A549, MCF7, PC3, HT29, HA1E.')

parser.add_argument('--ablation_mode', type=str, default='full', 
    choices=['full'],
    help="Ablation mode to analyze model components.")
parser.add_argument('--uniprot_embed_method', type=str, default='esm2', 
    choices=['esm2', 'protbert', 'gene2vec'],
    help="Method for protein (Uniprot) embedding, choose from 'esm2', 'protbert', or 'gene2vec'. This determines how protein sequences are represented in the model.")

parser.add_argument('--device_num', type=int, default=0, 
    help='GPU device number to use for training (e.g., 0, 1, ...).')

parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility of results.')
parser.add_argument('--max_num_epochs', type=int, default=40, help='Maximum number of training epochs.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for optimizer.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for regularization.')
parser.add_argument('--l2norm', type=float, default=1e-5, help='L2 regularization strength.')
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512,256,512],
                    help='Hidden layer dimensions of the neural network (e.g., 512 256 512 for 3 layers).')
parser.add_argument('--graph_node_embed_dim', type=int, default=32, help='Dimension of node embeddings in graph learning modules.')
parser.add_argument('--batch_size', type=int, default=250, help='Number of samples per training batch.')
parser.add_argument('--gat_num_heads', type=int, default=3, help='Number of attention heads in Graph Attention Network (GAT).')
parser.add_argument('--early_stopping_patience', type=int, default=5, help='Number of epochs with no improvement before early stopping.')
parser.add_argument('--gene_space', type=str, default='best_inferred', help="Gene set to use for modeling (e.g., 'best_inferred', 'all', or custom set name).")

args = parser.parse_args()
args.device = f'cuda:{args.device_num}' if torch.cuda.is_available() else 'cpu'
args.hidden_dims = [int(dim) for dim in args.hidden_dims]

datadir = f'./data'           # can be downloaded from google drive link in github repo
outputdir = f'./output/comprehensive_TDEM/{args.cell}'

if not os.path.exists(datadir):
    raise FileNotFoundError(f"Data directory not found: {datadir}, first download the data from the provided link in the GitHub repository.")

if os.path.exists(outputdir):
    print(f"Output directory {outputdir} already exists. Overwriting existing files.")

if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    print(f"Created output directory: {outputdir}")

hyperparam_file = f'{outputdir}/Hyperparameter.txt'
loss_file = f'{outputdir}/losses.txt'
best_model_file = f'{outputdir}/best_model.pt'

# set the random seed 
torch.manual_seed(args.random_seed)

print('Start learning DeepTE model...')
print('Arguments:')
print(args)

###========================================================
### LOAD DATA 
###========================================================

de = pd.read_pickle(f'{datadir}/CMAP_03_level5_trp_cp_{args.cell}_known_target_cid.pkl')
de.pubchem_cid = de.pubchem_cid.astype(int)
de = de[de.qc_pass == 1]

#--------------------------------------------------------
if args.uniprot_embed_method == 'protbert':
    uniprot2embed_df = pd.read_csv(f'{datadir}/uniprot_longest_sequence_protbert_embedding.tsv.gz', sep='\t', compression = 'gzip', index_col = 0)
elif args.uniprot_embed_method == 'esm2':
    uniprot2embed_df = pd.read_csv(f'{datadir}/uniprot_ESM2_embeddings.tsv.gz', sep = '\t', compression = 'gzip', index_col = 0)
elif args.uniprot_embed_method == 'gene2vec':
    uniprot2embed_df = pd.read_pickle(f'{datadir}/uniprot_gene2vec.pkl')
else:
    raise ValueError(f"Unknown uniprot embedding method: {args.uniprot_embed_method}")


dti = pd.read_csv(f'{datadir}/all_DTI.tsv', sep='\t')
DTI = dti.pivot(index = 'CID', columns = 'UniProt', values = 'Binding').fillna(0).astype(int)
dt, de = match_space_CMAP(DTI, de,  min_cut = 0, max_cut = 110)
de['pubchem_cid_index'] = (de.groupby('pubchem_cid').cumcount() + 1).astype(str).radd(de['pubchem_cid'].astype(str) + "_")

# dimension of the data 
compl = list(set(de.pubchem_cid) & set(dt.index))
targetl = list(set(dt.columns) & set(uniprot2embed_df.index))

print('%s targets are dropped due to no data' % (set(dt.columns) - set(uniprot2embed_df.index)))

genel = set(de.columns[de.columns.str.match(r'^\d+$')])

args.target_dim = len(targetl)
args.comp_dim = len(compl)
args.target_embed_dim = uniprot2embed_df.shape[1] 

uniprot2embed_df = uniprot2embed_df.loc[targetl]
uniprot2embed = {idx: np.array(row) for idx, row in uniprot2embed_df.iterrows()}
target_to_index = {value: index for index, value in enumerate(targetl)}

#--------------------------------------------------------
cmap_ginfo = pd.read_csv(datadir + '/CMAP_ginfo.csv', index_col=0)

# Mapping gene_space to required parameters
gene_space_map = {
    'landmark': lambda: cmap_ginfo[cmap_ginfo.feature_space == 'landmark'].index.astype(str),
    'best_inferred': lambda: cmap_ginfo[cmap_ginfo.feature_space != 'inferred'].index.astype(str),
}

gene_list = gene_space_map[args.gene_space]()
cid2exp = {row['pubchem_cid_index']: np.array(row.loc[gene_list], dtype=np.float32) for _, row in de.iterrows()}
args.gene_dim = len(gene_list) 
args.genel = gene_list

#--------------------------------------------------------
ent2id = {gene: idx for idx, gene in enumerate(args.genel)}
string_df = pd.read_csv(f'{datadir}/string_coexpression_score_ENTREZ_mapped.csv')
string_df['ens1'] = string_df.ent1.astype(str).map(ent2id)
string_df['ens2'] = string_df.ent2.astype(str).map(ent2id)

string_df_f = string_df.loc[:, ['ens1', 'ens2']].dropna()

args.ppi_edge_data = string_df_f

# graph_data = create_graph_data(string_df_f)
graph_data = create_graph_data_without_score(string_df_f)

args.graph_data = graph_data

#--------------------------------------------------------
print('Data loaded...')
save_model_parameters(hyperparam_file, args, parser)

###========================================================
### LOAD DATA 
###========================================================

dti = dti[dti.CID.isin(de.pubchem_cid) & dti.UniProt.isin(targetl)]
dti = pd.merge(dti, de.loc[:, ['pubchem_cid', 'pubchem_cid_index']], 
    left_on = 'CID', right_on = 'pubchem_cid', how = 'left')

dti['DE'] = dti.pubchem_cid_index.map(cid2exp)
dti['UniProt_embed'] = dti.UniProt.map(uniprot2embed)

train_pos = dti
train_pos.to_pickle(f'{outputdir}/train_dti.pkl')
#--------------------------------------------------------

train_dataset = DTI_dataset_with_sampled_true_neg(args, train_pos)
train_dataloader = DataLoader(train_dataset, batch_size=round(args.batch_size / 2), shuffle=True)

with open(hyperparam_file, "a") as f:
    f.write(f"Train dataset size: {len(train_dataloader.dataset)}\n")

###========================================================
# LEARNING
###========================================================

target2exp = generate_TE(args).to(args.device)

optimizer = torch.optim.Adam(target2exp.parameters(), lr=args.lr, weight_decay=args.l2norm)
criterion = nn.BCEWithLogitsLoss().to(args.device)
similarity_calculator = SimilarityScore(alpha=0.8, beta=0.2).to(args.device)

train_losses = []
best_test_loss = float('inf')
no_improvement_count = 0

print("Training started...")
for epoch in range(args.max_num_epochs): 
    target2exp.train()
    total_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.max_num_epochs}', leave=False):
        positive = batch['positive']
        negative = batch['negative']
        # Get positive and negative samples
        UniProt_embed_pos = positive['UniProt_embed'].float().to(args.device)
        drug_exp_pos = positive['DE'].to(args.device)
        label_pos = positive['label'].to(args.device)
        cid_pos = positive['CID']
        uniprot_pos = positive['UniProt']
        # Get negative samples
        UniProt_embed_neg = negative[0]['UniProt_embed'].float().to(args.device)
        drug_exp_neg = negative[0]['DE'].to(args.device)
        label_neg = negative[0]['label'].to(args.device)
        cid_neg = [neg['CID'] for neg in negative]  
        uniprot_neg = [neg['UniProt'] for neg in negative]
        output_pos = target2exp(UniProt_embed_pos)
        output_neg = target2exp(UniProt_embed_neg)
        score_pos = similarity_calculator(output_pos, drug_exp_pos)
        score_neg = similarity_calculator(output_neg, drug_exp_neg)        
        logits = torch.cat((score_pos, score_neg), dim=0)
        labels = torch.cat((label_pos,label_neg),dim=0)
        loss = criterion(logits,labels)  
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_loss)


print('Training completed...')

torch.save(target2exp.state_dict(), best_model_file)
print('Saved the model...')

###========================================================
# SAVE THE RESULTS
###========================================================

print('Generation of Comprehensive TDEM started...')

# Convert data to a PyTorch tensor
uniprot2gene = pd.read_csv(f'{datadir}/uniprot2geneID.tsv', sep = '\t', index_col=0)

if args.uniprot_embed_method == 'protbert':
    uniprot2embed_df = pd.read_csv(f'{datadir}/uniprot_longest_sequence_protbert_embedding.tsv.gz', sep='\t', compression = 'gzip', index_col = 0)
elif args.uniprot_embed_method == 'esm2':
    uniprot2embed_df = pd.read_csv(f'{datadir}/uniprot_ESM2_embeddings.tsv.gz', sep = '\t', compression = 'gzip', index_col = 0)
elif args.uniprot_embed_method == 'gene2vec':
    uniprot2embed_df = pd.read_pickle(f'{datadir}/uniprot_gene2vec.pkl')
else:
    raise ValueError(f"Unknown uniprot embedding method: {args.uniprot_embed_method}")

uniprot2embed_df = uniprot2embed_df.loc[uniprot2embed_df.index.isin(uniprot2gene.uniprot_id)]
all_data = torch.tensor(uniprot2embed_df.values, dtype=torch.float32)

dataset = TensorDataset(all_data)
data_loader = DataLoader(dataset, batch_size=1000)

# Process batches
all_results = []  # To collect the results

with torch.no_grad():
    for batch in tqdm(data_loader):
        batch_data = batch[0].to(args.device)  # Move to device
        batch_results = target2exp(batch_data)  # Process batch
        all_results.append(batch_results)

# Combine results back into a single tensor
final_result = torch.cat(all_results, dim=0)
final_result_df = pd.DataFrame(final_result.cpu().numpy())
final_result_df.columns = gene_list
final_result_df['UniProt'] = uniprot2embed_df.index
final_result_df.to_csv(f'{outputdir}/All_uniprot_TE_matrix.csv', index=False)

print('TDEM matrix saved...')
print('Done!')