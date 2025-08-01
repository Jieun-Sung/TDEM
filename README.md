# TDEM
We introduce TDEM (Target-
induced Differential Expression Matrix), a novel computational framework that generates in silico target-perturbed
expression profiles from drug-induced transcriptomic data. TDEM circumvents experimental noise by learning target
representations computationally rather than relying on confounded genetic perturbation data. 

![Model architecture](./figures/scheme_v1.png)

## Table of Contents

- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Usage](#usage)
- [Data](#data)
- [Citation](#citation)


## Installation

1. Clone the repository and download the `TDEM.yaml` environment file.
2. Create the conda environment:
    ```bash
    conda env create -f TDEM.yaml
    ```
3. Activate the environment:
    ```bash
    conda activate TDEM
    ```
    
## Data

All data used for model training, the pretrained models for six cell lines (A375, A549, HT29, HA1E, MCF7, and PC3), and the comprehensive TDEM dataset can be downloaded from the following link: 
[Google Drive](https://drive.google.com/drive/folders/1rnlX_vkhixhHDbbmGSW5WDtSUSMvndHc?usp=drive_link)

## Preprocessing

- **Extract protein sequences from UniProt IDs**
  - Script: `get_uniprot_sequences.py`
  - Usage:
    ```bash
    python get_uniprot_sequences.py \
        --input_seqfile /path/to/uniprot_id.csv \
        --prefix test \
        --output_dir /path/to/output_directory \
        --ncore 40
    ```
  - Input: CSV file with a column named `uniprot_id` or `UniProt`
  - Output: `<prefix>_uniprot2sequence.tsv` in the specified output directory

- **Embed protein sequences using Pre-trained Language Models (PLM)**
  - Script: `embed_sequence_with_PLM.py`
  - Usage:
    ```bash
    python embed_sequence_with_PLM.py \
        --input_seqfile /path/to/uniprot2sequence.tsv \
        --PLM esm2 \
        --prefix test \
        --output_dir /path/to/output_directory \
        --batch_size 8
    ```
  - Input: TSV file of protein sequences (output of previous step)
  - Output: `<prefix>_esm2_embeddings.tsv.gz` in the specified output directory



## Usage

[Add instructions for how to use your project, such as example commands, script usage, etc.]

## Citation

[If applicable, add citation information or links to relevant papers.]
