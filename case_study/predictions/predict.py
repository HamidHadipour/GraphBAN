
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import warnings
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaModel
import esm
from models import GraphBAN
from utils import set_seed, graph_collate_func, mkdir
from dataloader import DTIDataset, MultiDataLoader, DTIDataset2
from configs import get_cfg_defaults
from domain_adaptator import Discriminator
from trainer_pred import Trainer

def parse_paths(input_string):
    # Split input by comma or space
    return input_string.replace(',', ' ').split()

# Set up argument parser
parser = argparse.ArgumentParser(description="Load train, val, test datasets and additional parameters.")
parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
parser.add_argument("--folder_path", type=str, help='Path to the folder containing .pth files')

#parser.add_argument('--trained_models', type=parse_paths, help="List of file paths")
#parser.add_argument("--trained_model", type=str, required=True, help='path to the saved model.pth')
parser.add_argument("--save_dir", type=str, required=True, help='path to save the csv of prediction probabilities')

args = parser.parse_args()

# Read test dataset
df_test = pd.read_csv(args.test_path)
df_test['Protein'] = df_test['Protein'].apply(lambda x: x[:1022] if len(x) > 1022 else x)
print('shape of your dataset:', df_test.shape)

# Setup the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ESM model for protein embeddings
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
esm_model = esm_model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

# Function to get protein features
def Get_Protein_Feature(p_list):
    data_tmp = [(f"protein{i}", p[:1022]) for i, p in enumerate(p_list)]
    dictionary = {}
    for i in range((len(data_tmp) + 4) // 5):  # Process in chunks of 5
        data_part = data_tmp[i * 5:(i + 1) * 5]
        _, _, batch_tokens = batch_converter(data_part)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        for j, (_, seq) in enumerate(data_part):
            emb_rep = token_representations[j, 1:len(seq) + 1].mean(0).cpu().numpy()
            dictionary[seq] = emb_rep
    return pd.DataFrame(dictionary.items(), columns=['Protein', 'esm'])

# Merge ESM features
pro_list_test = df_test['Protein'].unique()
df_test = pd.merge(df_test, Get_Protein_Feature(list(pro_list_test)), on='Protein', how='left')

# Load a pretrained ChemBERTa model and tokenizer for SMILES embeddings
model_name = "DeepChem/ChemBERTa-77M-MTR"
model_chem = RobertaModel.from_pretrained(model_name, num_labels=2, add_pooling_layer=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to get ChemBERTa embeddings
def get_embeddings(df):
    emb_list = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):
        encodings = tokenizer(row['SMILES'], return_tensors='pt', padding="max_length", max_length=290, truncation=True).to(device)
        with torch.no_grad():
            output = model_chem(**encodings)
            smiles_embeddings = output.last_hidden_state[0, 0].cpu().numpy()
            emb_list.append(smiles_embeddings)
    return emb_list

# Merge SMILES features and remove duplicate rows
df_test_unique = df_test.drop_duplicates(subset='SMILES')
emb_list_test = get_embeddings(df_test_unique)
df_test_unique['fcfp'] = emb_list_test
df_test = pd.merge(df_test, df_test_unique[['SMILES', 'fcfp']], on='SMILES', how='left')

# Configurations and model setup
cfg = get_cfg_defaults()
cfg.merge_from_file("GraphBAN_DA.yaml")
cfg.freeze()
mkdir(args.folder_path)
# Set up the DataLoader
test_dataset = DTIDataset(df_test.index.values, df_test)
test_generator = DataLoader(test_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=cfg.SOLVER.NUM_WORKERS, drop_last=False, collate_fn=graph_collate_func)

# Load GraphBAN model and optimizer
modelG = GraphBAN(**cfg).to(device)
opt = torch.optim.Adam(modelG.parameters(), lr=cfg.SOLVER.LR)

import os
import argparse

def get_pth_files(folder_path):
    # Get all .pth files in the folder
    pth_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pth')]
    return pth_files[30:51]
folder_path = args.folder_path
pth_files = get_pth_files(folder_path)
i = 30
for model in pth_files:
    
	
# Load trained model and train
     modelG.load_state_dict(torch.load(model))
     trainer = Trainer(modelG, opt, device, test_generator, **cfg)
     pred = trainer.train()
     print(i)
     

# Save results
     df_test[f'pred{i}'] = pred
     i+=1
del df_test['esm']
del df_test['fcfp']
smiles = df_test['SMILES']
proteins = df_test['Protein']
del df_test['SMILES']
del df_test['Protein']
del df_test['Y']
df_test['row_average'] = df_test.mean(axis=1)

new_data = pd.DataFrame()
new_data['SMILES'] = smiles
new_data['Protein'] = proteins

new_data['predicted_value'] = df_test['row_average']

new_data.to_csv(args.folder_path+'/' + args.save_dir, index=False)

#print("\nThe prediction probabilities saved in result/" + args.save_dir + '\n')
#print("The prediction scores saved in result/"+"test_markdowntable_of_prediction_with_trained_model.txt")
