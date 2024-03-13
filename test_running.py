from models import GraphBAN
from time import time
from utils import set_seed, graph_collate_func, mkdir,graph_collate_func2
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader, DTIDataset2
from torch.utils.data import DataLoader
from trainer import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import torch
import torch.nn as nn
import copy
import os
import numpy as np
from tqdm import tqdm
from rdkit.Chem import AllChem
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
import pandas as pd
from torch.nn.utils.weight_norm import weight_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# If you want to change the settings such as number of epochs for teh GraphBAN`s main model change it through GraphBAN_Demo.yaml.
# If you want to run the model for transductive analysis, use GraphBAN_None_DA.yaml
cfg_path = "/content/GraphBAN/GraphBAN_Demo.yaml"
#cfg_path = "GraphBAN_DA.yaml"
cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_path)
cfg.freeze()
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")
set_seed(cfg.SOLVER.SEED)
mkdir(cfg.RESULT.OUTPUT_DIR)
experiment = None
print(f"Config yaml: {cfg_path}")
print(f"Running on: {device}")
print(f"Hyperparameters:")
dict(cfg)

# Read your custom dataset here. it should be separated in three divitions for any of inductive or transductive analysis in the form of CSV or parquet.
df_train = pd.read_csv("/content/GraphBAN/Data/Demo_data/train.csv")
df_val = pd.read_csv("/content/GraphBAN/Data/Demo_data/val.csv")
df_test = pd.read_csv("/content/GraphBAN/Data/Demo_data/test.csv")

from rdkit.Chem import AllChem
import sys
import numpy as np
sys.path.append('/usr/local/lib/python3.7/site-packages/')

try:
  from rdkit import Chem
  from rdkit.Chem.Draw import IPythonConsole
except ImportError:
  print('Stopping RUNTIME. Colaboratory will restart automatically. Please run again.')
  exit()
df_list = [df_train,df_val, df_test]
for dfs in df_list:

    x_batch11 = []
    # y = torch.Tensor([y])
    smiles2 = dfs.iloc[dfs.index]['SMILES']
    batch_smiles2 = list(smiles2)
    for item in batch_smiles2:

        m1 = Chem.MolFromSmiles(str(item))
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius=2, nBits=1024)
        x = np.array(fp1, dtype=np.float64)
        x_batch11.append(x)
    dfs['fcfp'] = x_batch11

train_emb = pd.read_parquet("Data/Demo_data/Demo_teacher_embedding.parquet")

train_emb['Array'] = train_emb.apply(lambda row: np.array(row), axis=1)

# Drop all columns except the 'Array' column
train_emb.drop(train_emb.columns.difference(['Array']), axis=1, inplace=True)

df_train['teacher_emb'] = train_emb['Array']

train_dataset = DTIDataset2(df_train.index.values, df_train)
val_dataset = DTIDataset(df_val.index.values, df_val)
test_dataset = DTIDataset(df_test.index.values, df_test)

params1 = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS, 'drop_last': True, 'collate_fn': graph_collate_func}
params2 = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS, 'drop_last': True, 'collate_fn': graph_collate_func2}
source_generator = DataLoader(train_dataset, **params2)
target_generator = DataLoader(val_dataset, **params1)
n_batches = max(len(source_generator), len(target_generator))
multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
training_generator = DataLoader(train_dataset, **params2)
params1['shuffle'] = False
params1['drop_last'] = False
val_generator = DataLoader(val_dataset,**params1)
test_generator = DataLoader(test_dataset,**params1)

model = GraphBAN(**cfg).to(device)
opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
if torch.cuda.is_available():
  torch.backends.cudnn.benchmark = True

# In the case that you need to run inductive analysis teh cfg.DA.USE is True otherwise you will run transductive analysis

if cfg.DA.USE:
        if cfg["DA"]["RANDOM_LAYER"]:
            domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
        else:
            domain_dmm = Discriminator(input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
                                       n_class=cfg["DECODER"]["BINARY"]).to(device)
        # params = list(model.parameters()) + list(domain_dmm.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
torch.backends.cudnn.benchmark = True

trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None, discriminator=None, experiment=None, **cfg)
result = trainer.train()

if __name__ == "__main__":
    print("running started")
