import os,sys

from utils import *

###========================================================
### ARGUMENTS 
###========================================================

parser = argparse.ArgumentParser(description='Train and evaluate the model for cell line gene expression prediction.')

parser.add_argument('--cell', type=str, required=True, 
        choices=['A375', 'A549', 'MCF7', 'PC3', 'HT29', 'HA1E'],
        help='Cell line to use for training and evaluation. Choose from: A375, A549, MCF7, PC3, HT29, HA1E.')

parser.add_argument('--ablation_mode', type=str, default='full', 
    choices=['full', 'no_gat', 'no_pretrained', 'no_correlation', 'no_dot_product'],
    help="Ablation mode to analyze model components. 'full': use all modules, 'no_gat': remove GAT module, 'no_pretrained': don't use pretrained embedding, 'no_correlation': remove correlation features, 'no_dot_product': remove dot product features.")

parser.add_argument('--uniprot_embed_method', type=str, default='esm2', 
    choices=['esm2', 'protbert', 'gene2vec'],
    help="Method for protein (Uniprot) embedding, choose from 'esm2', 'protbert', or 'gene2vec'. This determines how protein sequences are represented in the model.")

parser.add_argument('--device_num', type=int, default=0, 
    help='GPU device number to use for training (e.g., 0, 1, ...).')

parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility of results.')
parser.add_argument('--max_num_epochs', type=int, default=300, help='Maximum number of training epochs.')
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
outputdir = f'./output/benchmark/TDEM_{args.uniprot_embed_method}/{args.cell}'

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

hyperparam_file = f'{outputdir}/Hyperparameter.txt'
loss_file = f'{outputdir}/losses.txt'
best_model_file = f'{outputdir}/best_model.pt'

# set the random seed 
torch.manual_seed(args.random_seed)

print('Start learning TDEM model...')
print('Arguments:')
print(args)

###========================================================
### LOAD DATA 
###========================================================

de = pd.read_pickle(f'{datadir}/CMAP_03_level5_trp_cp_{args.cell}_known_target_cid.pkl')
de.pubchem_cid = de.pubchem_cid.astype(int)
de = de[de.qc_pass == 1]

#--------------------------------------------------------
# DTI matched with other benchmark tools (SSGCN, FRoGS)

train_dti = pd.read_csv(f'{datadir}/benchmark_DTI_dataset/{args.cell}/TDEM_train_dti.tsv', sep = '\t')
val_dti = pd.read_csv(f'{datadir}/benchmark_DTI_dataset/{args.cell}/TDEM_val_dti.tsv', sep = '\t')
test_dti = pd.read_csv(f'{datadir}/benchmark_DTI_dataset/{args.cell}/TDEM_test_dti.tsv', sep = '\t')
dti = pd.concat([train_dti, val_dti, test_dti], axis=0).reset_index(drop=True)

if args.uniprot_embed_method == 'protbert':
    uniprot2embed_df = pd.read_csv(f'{datadir}/uniprot_longest_sequence_protbert_embedding.tsv.gz', sep='\t', compression = 'gzip', index_col = 0)
elif args.uniprot_embed_method == 'esm2':
    uniprot2embed_df = pd.read_csv(f'{datadir}/uniprot_ESM2_embeddings.tsv.gz', sep = '\t', compression = 'gzip', index_col = 0)
elif args.uniprot_embed_method == 'gene2vec':
    uniprot2embed_df = pd.read_pickle(f'{datadir}/uniprot_gene2vec.pkl')
else:
    raise ValueError(f"Unknown uniprot embedding method: {args.uniprot_embed_method}")

#--------------------------------------------------------
DTI = dti.pivot(index = 'CID', columns = 'UniProt', values = 'Binding').fillna(0).astype(int)
dt = DTI
de['pubchem_cid_index'] = (de.groupby('pubchem_cid').cumcount() + 1).astype(str).radd(de['pubchem_cid'].astype(str) + "_")

# dimension of the data 
compl = list(set(de.pubchem_cid) & set(DTI.index))
targetl = list(set(DTI.columns) & set(uniprot2embed_df.index))

print('%s targets are dropped due to no data' % (set(dt.columns) - set(uniprot2embed_df.index)))
genel = set(de.columns[de.columns.str.match(r'^\d+$')])

args.target_dim = len(targetl)
args.comp_dim = len(compl)
args.target_embed_dim = uniprot2embed_df.shape[1] 

uniprot2embed_df = uniprot2embed_df.loc[targetl].dropna()
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

train_pos = dti[dti.CID.isin(pd.read_csv(f'{datadir}/benchmark_DTI_dataset/{args.cell}/train_CID.txt', header = None).iloc[:,0])]
val_pos = dti[dti.CID.isin(pd.read_csv(f'{datadir}/benchmark_DTI_dataset/{args.cell}/val_CID.txt', header = None).iloc[:,0])]
test_pos = dti[dti.CID.isin(pd.read_csv(f'{datadir}/benchmark_DTI_dataset/{args.cell}/test_CID.txt', header = None).iloc[:,0])]

train_dataset = DTI_dataset_with_sampled_true_neg(args, train_pos)
train_dataloader = DataLoader(train_dataset, batch_size=round(args.batch_size / 2), shuffle=True)

val_dataset = DTI_dataset_with_sampled_true_neg(args, val_pos)
val_dataloader = DataLoader(val_dataset, batch_size=round(args.batch_size / 2), shuffle=False)

test_dataset = DTI_dataset_for_prediction(args, test_pos)
test_dataloader = DataLoader(test_dataset, batch_size=round(args.batch_size / 2), shuffle=False)

with open(hyperparam_file, "a") as f:
    f.write(f"Train dataset size: {len(train_dataloader.dataset)}\n")
    f.write(f"Validation dataset size: {len(val_dataloader.dataset)}\n")
    f.write(f"Test dataset size: {len(test_dataloader.dataset)}\n")

#--------------------------------------------------------
# SAVE THE DATA
#--------------------------------------------------------

train_pos.to_pickle(f'{outputdir}/train_pos.pkl')
val_pos.to_pickle(f'{outputdir}/val_pos.pkl')
test_pos.to_pickle(f'{outputdir}/test_pos.pkl')

de.to_pickle(f'{outputdir}/DE_matrix.pkl')
dti.to_pickle(f'{outputdir}/DTI_with_DE_uniprotembed.pkl')
cid2exp_filepath = f'{outputdir}/cid2exp.pkl'

with open(cid2exp_filepath, 'wb') as f:
    pickle.dump(cid2exp, f)

with open(f'{outputdir}/graph_data.pkl', 'wb') as f:
    pickle.dump(graph_data, f)

# save train_dataloader, val_dataloader, test_dataloader
with open(f'{outputdir}/dataloaders.pkl', 'wb') as f:
    pickle.dump([train_dataloader, val_dataloader, test_dataloader], f)


#--------------------------------------------------------
# LEARNING
#--------------------------------------------------------

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
    # Validation phase
    target2exp.eval()
    total_test_loss = 0.0
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{args.max_num_epochs}', leave=False):
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
            test_loss = criterion(logits,labels)  
            total_test_loss += test_loss.item()
            # Collect labels and probabilities
            probabilities = torch.sigmoid(logits).cpu().numpy()
            labels = labels.cpu().numpy()
            all_labels.extend(labels)
            all_probabilities.extend(probabilities)
    avg_test_loss = total_test_loss / len(val_dataloader)
    metrics = calculate_metrics_during_training(all_labels, all_probabilities)
    print(f"Epoch {epoch + 1}/{args.max_num_epochs}: Train loss: {avg_loss:.4f}, Val loss: {avg_test_loss:.4f}, AUROC: {metrics['AUROC']:.4f}, AUPRC: {metrics['AUPRC']:.4f}, F1_Score: {metrics['F1_Score']:.4f}, Top-100 Accuracy: {metrics['Top_100_Accuracy']:.4f}, Top-30 Accuracy: {metrics['Top_30_Accuracy']:.4f}")
    # Save loss to file
    with open(loss_file, "a") as f:
        f.write(f"{epoch}\t{avg_loss}\t{avg_test_loss}\t{metrics['AUROC']}\t{metrics['AUPRC']}\t{metrics['F1_Score']}\t{metrics['Top_100_Accuracy']}\t{metrics['Top_30_Accuracy']}\n")
    # Save the best model based on test loss
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        no_improvement_count = 0
        torch.save(target2exp.state_dict(), best_model_file)
        print(f"New best model saved with test loss: {best_test_loss}")
    else:
        no_improvement_count += 1
    # Early stopping
    if no_improvement_count >= args.early_stopping_patience:
        final_epoch = epoch - args.early_stopping_patience + 1
        print(f"Stopping early as there has been no improvement for {args.early_stopping_patience} epochs. Trained to {final_epoch} epochs.")
        # Log early stopping info to a file
        with open(hyperparam_file, "a") as f:
            f.write(f"Training stopped at epoch {final_epoch} due to early stopping\n")
        break


#--------------------------------------------------------
# EVALUATION - F1 SCORE, AUROC, AUPRC, ACCURACY
#--------------------------------------------------------

target2exp = generate_TE(args)
target2exp.load_state_dict(torch.load(best_model_file, map_location=args.device))
target2exp.to(args.device)
target2exp.eval()

all_labels = []
all_probabilities = []
total_test_loss = 0.0

res_in_dataframe = pd.DataFrame()

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        dt = batch
        UniProt_embed_pos = dt['UniProt_embed'].float().to(args.device)
        drug_exp_pos = dt['DE'].to(args.device)
        label_pos = dt['label'].to(args.device)
        cid_pos = dt['CID']
        uniprot_pos = dt['UniProt']
        output_pos = target2exp(UniProt_embed_pos)
        score_pos = similarity_calculator(output_pos, drug_exp_pos)
        logits = score_pos
        labels = label_pos
        test_loss = criterion(logits,labels)
        total_test_loss += test_loss.item()
        probabilities = torch.sigmoid(logits).cpu().numpy()
        labels = labels.cpu().numpy()
        all_labels.extend(labels)
        all_probabilities.extend(probabilities)
        pos_cids = [cid.item() for cid in cid_pos]
        pos_uniprots = [uniprot for uniprot in uniprot_pos]
        batch_df = pd.DataFrame({
            'prediction': logits.cpu().detach().numpy(),
            'label': labels,
            'CID': pos_cids,
            'UniProt': pos_uniprots
        })
        res_in_dataframe = pd.concat([res_in_dataframe, batch_df], ignore_index=True)



all_probabilities = np.array(all_probabilities)
all_labels = np.array(all_labels)

np.save(f'{outputdir}/all_probabilities.npy', all_probabilities)
np.save(f'{outputdir}/all_labels.npy', all_labels)
res_in_dataframe.to_csv(f'{outputdir}/pred_true_value_with_CID_UNIPROT_pair.csv', index=False)


# Compute the metrics
auroc = roc_auc_score(all_labels, all_probabilities)
auprc = average_precision_score(all_labels, all_probabilities)

# Convert logits to binary predictions (threshold at 0.5)
predicted_labels = (all_probabilities >= 0.5).astype(int)

# Calculate F1 score and accuracy
f1 = f1_score(all_labels, predicted_labels)
accuracy = accuracy_score(all_labels, predicted_labels)

# Print the results
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

result_file = f'{outputdir}/results.txt'
with open(result_file, "w") as f:
    f.write(f"AUROC: {auroc:.4f}\n")
    f.write(f"AUPRC: {auprc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")


fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
precision, recall, _ = precision_recall_curve(all_labels, all_probabilities)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.text(0.6, 0.3, f"AUROC = {auroc:.4f}", fontsize=12)
plt.plot([0, 1], [0, 1], linestyle="--", color="black")

plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.text(0.6, 0.9, f"AUPRC = {auprc:.4f}", fontsize=12)

plt.tight_layout()
plt.savefig(f'{outputdir}/AUROC_PR_curves.png')

