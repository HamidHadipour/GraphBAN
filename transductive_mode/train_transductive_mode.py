import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch.nn import Linear, Dropout
import torch.optim as optim
from torchmetrics import AUROC
from torchmetrics import AUROC, AveragePrecision, F1Score
import argparse
import torch_geometric.transforms as T
from torch import nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
# Import other necessary libraries here...

# Set up argument parser
parser = argparse.ArgumentParser(description="Load train, val, test datasets and additional parameters.")

parser.add_argument("--train_path", type=str, required=True, help="Path to the train dataset.")
parser.add_argument("--val_path", type=str, required=True, help="Path to the test dataset.")
parser.add_argument("--test_path", type=str, required=True, help="Path to the validation dataset.")
parser.add_argument("--seed", type=int, required=True, help="Seed number for random processes.")
parser.add_argument("--save_model", type=str, required=True, help="Path to the saved model.")
parser.add_argument("--metric_path", type=str, required=True, help="Path to save the predicted metrics.")
parser.add_argument("--prediction_path", type=str, required=True, help="Path to save the prediction probabilities.")
parser.add_argument("--h_dimension", type=int, required=True, help="hidden dimension.")
parser.add_argument("--epochs", type=int, required=True, help="number of training epochs.")

args = parser.parse_args()



import random
seed_value=args.seed
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)




def load_node_csv(path, index_col, encoders=None, **kwargs):

    df = pd.read_csv(path, index_col=index_col, **kwargs)

    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return  mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None, **kwargs):
  labels_test = []
  df = pd.read_csv(path, **kwargs)
  #df = path
  df_active = df.loc[df['Y']==1]

  src = [src_mapping[index] for index in df_active[src_index_col]]
  dst = [dst_mapping[index] for index in df_active[dst_index_col]]
  edge_index = torch.tensor([src, dst])#, dtype=torch.long).type(torch.LongTensor)

  labels_test = df["Y"]

  # a temporary list to store the string labels
  temp_list = df["Y"].tolist()


  edge_attr = torch.tensor(temp_list, dtype=torch.long)
  ############################################################
  src = [src_mapping[index] for index in df[src_index_col]]
  dst = [dst_mapping[index] for index in df[dst_index_col]]
  edge_label_index = torch.tensor([src, dst], dtype=torch.long).type(torch.LongTensor)




  return edge_index,edge_label_index, edge_attr

from torch.nn import Dropout
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), out_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.conv3 = SAGEConv((-1, -1), out_channels)
        self.conv4 = SAGEConv((-1, -1), out_channels)
        self.dropout = Dropout(p=0.2)
    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index).relu()

        x = self.conv2(x, edge_index).relu()

        x = self.conv3(x, edge_index).relu()
        #x = self.conv4(x, edge_index).relu()


        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.lin1 = Linear(2*hidden_channels,hidden_channels)
        self.lin2 = Linear(hidden_channels,hidden_channels)
        self.lin3 = Linear(hidden_channels,hidden_channels)

        self.lin4 = Linear(hidden_channels,1)

        self.dropout = Dropout(p=0.2)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        x = torch.cat([z_dict['chem'][row], z_dict['protein'][col]], dim=-1)
        z1 = self.lin1(x).relu()
        z1 = self.dropout(z1)
        z1 = self.lin2(z1).relu()
        z1 = self.dropout(z1)
        z1 = self.lin3(z1)
        z1 = self.dropout(z1)
        z1 = self.lin4(z1)
        z2 = z1.view(-1)

        return z2,x,edge_label_index


class Model(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        self.encoder1 = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder2 = to_hetero(self.encoder1, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)


    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder2(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

def train(model, optimizer, data):
    train_data_u = data
    model.train()
    optimizer.zero_grad()
    pred, emb, ed = model(train_data_u.x_dict, train_data_u.edge_index_dict,
                 train_data_u['chem', 'CPI', 'protein'].edge_label_index)
    target = train_data_u['chem', 'CPI', 'protein'].edge_label.float()
    BCEW = nn.BCEWithLogitsLoss()
    loss = BCEW(pred, target)
    loss.backward()
    optimizer.step()
    #print(loss)
    return  float(loss)

@torch.no_grad()
def test_auroc(model, data, test_df):
    model.eval()
    pred,emb,ed = model(data.x_dict, data.edge_index_dict,
                 data['chem', 'CPI', 'protein'].edge_label_index)
    target = data['chem', 'CPI', 'protein'].edge_label#.float()
    target = target.to(torch.long)
    pred = torch.sigmoid(pred)
    #target = torch.tensor(target, dtype = torch.long)
    auprc = AveragePrecision(task='binary')
    auprc = auprc(pred, target)
    auroc = AUROC(task = 'binary')
    aurocscore = auroc(pred, target)

    # AUPRC (Average Precision)
    
    #auprc = average_precision_score(target, pred)
    

# F1 Score

    f1 = F1Score(task='binary')
    f1_score = f1(pred, target)
    #pred = list(pred)
    pred = pred.cpu().numpy()
    #target = np.array(target)
    #target = target.astype(int)
    #fpr, tpr, thresholds = roc_curve(pred, target)
    #prec, recall, _ = precision_recall_curve(pred, target)
    #precision = tpr / (tpr + fpr)
    #f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    test_df222 = pd.DataFrame()
    #pred = pred.cpu().numpy()
    test_df222['pred'] = pred
    test_df222['target'] = target
    return aurocscore,auprc, f1_score, test_df222

def run_model(data_file_path_train, mol_feature_train, prot_feature_train,data_file_path_val, mol_features_stan_val2,protein_features_stan_val2, data_file_path_test, mol_feature_test, prot_feature_test ):
  test_df = pd.read_csv(data_file_path_test)
  #set_seeds()
  cp_path = data_file_path_train
  cp_mapping_train = load_node_csv(
    cp_path, index_col='SMILES'
    )
  cp_mapping_p_train = load_node_csv(
    cp_path, index_col='Protein'
    )
  edge_index_train,edge_label_index_train, edge_label_train = load_edge_csv(
    cp_path,
    src_index_col='SMILES',
    src_mapping=cp_mapping_train,
    dst_index_col='Protein',
    dst_mapping=cp_mapping_p_train
    )

  data = HeteroData()
  data['chem'].num_nodes =len(cp_mapping_train)
  data['protein'].num_nodes =len(cp_mapping_p_train)
  data['chem', 'CPI', 'protein'].edge_index = edge_index_train.type(torch.int64)
  data['chem'].x = torch.tensor(mol_feature_train).to(torch.float32)#torch.eye(data['chem'].num_nodes,128)
  del data['chem'].num_nodes
  data['protein'].x = torch.tensor(prot_feature_train).to(torch.float32)#torch.eye(data['protein'].num_nodes,128)
  del data['protein'].num_nodes
  data['chem', 'CPI', 'protein'].edge_label = edge_label_train
  data['chem', 'CPI', 'protein'].edge_label_index = edge_label_index_train
  data = T.ToUndirected()(data)
  del data['protein', 'rev_CPI', 'chem'].edge_label
  print(data)

      #set_seeds()
  cp_path = data_file_path_val
  cp_mapping_val = load_node_csv(
    cp_path, index_col='SMILES'
    )
  cp_mapping_p_val = load_node_csv(
    cp_path, index_col='Protein'
    )
  edge_index_val,edge_label_index_val, edge_label_val = load_edge_csv(
    cp_path,
    src_index_col='SMILES',
    src_mapping=cp_mapping_val,
    dst_index_col='Protein',
    dst_mapping=cp_mapping_p_val
    )

  data_val = HeteroData()
  data_val['chem'].num_nodes =len(cp_mapping_val)
  data_val['protein'].num_nodes =len(cp_mapping_p_val)
  data_val['chem', 'CPI', 'protein'].edge_index = edge_index_val.type(torch.int64)
  data_val['chem'].x = torch.tensor(mol_features_stan_val2).to(torch.float32)#torch.eye(data['chem'].num_nodes,128)
  del data_val['chem'].num_nodes
  data_val['protein'].x = torch.tensor(protein_features_stan_val2).to(torch.float32)#torch.eye(data['protein'].num_nodes,128)
  del data_val['protein'].num_nodes
  data_val['chem', 'CPI', 'protein'].edge_label = edge_label_val
  data_val['chem', 'CPI', 'protein'].edge_label_index = edge_label_index_val
  data_val = T.ToUndirected()(data_val)
  del data_val['protein', 'rev_CPI', 'chem'].edge_label
  print(data_val)


    #set_seeds()
  cp_path = data_file_path_test
  cp_mapping_test = load_node_csv(
    cp_path, index_col='SMILES'
    )
  cp_mapping_p_test = load_node_csv(
    cp_path, index_col='Protein'
    )
  edge_index_test,edge_label_index_test, edge_label_test = load_edge_csv(
    cp_path,
    src_index_col='SMILES',
    src_mapping=cp_mapping_test,
    dst_index_col='Protein',
    dst_mapping=cp_mapping_p_test
    )

  data_test = HeteroData()
  data_test['chem'].num_nodes =len(cp_mapping_test)
  data_test['protein'].num_nodes =len(cp_mapping_p_test)
  data_test['chem', 'CPI', 'protein'].edge_index = edge_index_test.type(torch.int64)
  data_test['chem'].x = torch.tensor(mol_feature_test).to(torch.float32)#torch.eye(data['chem'].num_nodes,128)
  del data_test['chem'].num_nodes
  data_test['protein'].x = torch.tensor(prot_feature_test).to(torch.float32)#torch.eye(data['protein'].num_nodes,128)
  del data_test['protein'].num_nodes
  data_test['chem', 'CPI', 'protein'].edge_label = edge_label_test
  data_test['chem', 'CPI', 'protein'].edge_label_index = edge_label_index_test
  data_test = T.ToUndirected()(data_test)
  del data_test['protein', 'rev_CPI', 'chem'].edge_label
  print(data_test)






    # Initialize and prepare the model, data, etc.
  hidden_channels= args.h_dimension
  model = Model(data, hidden_channels=hidden_channels)
  #model.load_state_dict(torch.load(args.trained_model))
  with torch.no_grad():
    model.encoder2(data.x_dict, data.edge_index_dict)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

  #train_rmse, auprc, f1 = test_auroc(model, data_test)
  #print(train_rmse, auprc,f1 )

  best_val_loss = float(0)
  for epoch in range(1, args.epochs):

    loss = train(model, optimizer, data)
    model.eval()
    val_auprc = 0.0
       
    pred,emb,ed = model(data_val.x_dict, data_val.edge_index_dict,
                 data_val['chem', 'CPI', 'protein'].edge_label_index)
    target = data_val['chem', 'CPI', 'protein'].edge_label#.float()
    target = target.to(torch.long)
    #target = torch.tensor(target, dtype = torch.long)
    auprc = AveragePrecision(task='binary')
    val_auprc = auprc(pred, target)
    #scheduler.step(val_auprc)
    if val_auprc > best_val_loss:
        best_val_loss = val_auprc
        torch.save(model.state_dict(), args.save_model)
        print("Model improved and saved.")
    else:
        print("Validation loss did not improve.")

    train_rmse, auprc, f1, test_df1 = test_auroc(model, data_val, test_df)
    print(epoch)
    print(val_auprc)
    # Create a DataFrame with the metrics
  model.load_state_dict(torch.load(args.save_model))
  train_rmse, auprc, f1, df_pred = test_auroc(model, data_test, test_df)
  print(train_rmse, auprc,f1 )  
  metrics_df = pd.DataFrame({
  'AUROC': [train_rmse],
  'AUPRC': [auprc],
  'F1 Score': [f1]
   })
  df_pred.to_csv(args.prediction_path, index = False)

# Save the DataFrame to a CSV file
  metrics_df.to_csv(args.metric_path,index=False)

  print("Metrics saved to metrics.csv")

  torch.save(model.state_dict(), args.save_model)
#save link embeddings in a parquet file
 # emb_np = emb.detach().cpu().numpy()
  #emb_df = pd.DataFrame(emb_np)
  #embcol = []
  #for i in range(hidden_channels*2):
   # embcol.append(str(i))

  #emb_df.columns = embcol
  #emb_df.to_parquet('pdb20_transductive_teacher_emb.parquet')
  #message = f"Saved embeddings to {output_file_path}"
  
  return 'yes'
import torch
import esm
import pandas as pd
import numpy as np
path_train = args.train_path#'Data/biosnap/transductive/seed14/train_biosnap14.csv'
path_val = args.val_path
path_test = args.test_path#'Data/biosnap/transductive/seed14/test_biosnap14.csv'
df_train = pd.read_csv(path_train)
df_val = pd.read_csv(path_val)
df_test = pd.read_csv(path_test)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

esm_model = esm_model.eval().to(device)


def Get_Protein_Feature(p_list):
    feature = []
    data_tmp = []
    dictionary = {}
    i = 0
    for p in p_list:
        p = p[0:1022]
        data_tmp.append(("protein" + str(i), p))
        i = i + 1
    # print(len(data_tmp))

    sequence_representations = []

    for i in range(len(data_tmp) // 5 + 1):
        # print(i)
        if i == len(data_tmp) // 5:
            data_part = data_tmp[i * 5:]
        else:
            data_part = data_tmp[i * 5:(i + 1) * 5]

        if not data_part:  # Check if data_part is empty
            continue

        data_part = [(label, sequence) for label, sequence in data_part]
        _, _, batch_tokens = batch_converter(data_part)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        for j, (_, seq) in enumerate(data_part):
            emb_rep = token_representations[j, 1:len(seq) + 1].mean(0)
            emb_rep = emb_rep.cpu().numpy()
            # sequence_representations.append(emb_rep.cpu().numpy())
            dictionary[seq] = emb_rep
            df = pd.DataFrame(dictionary.items(), columns=['Protein', 'esm'])
            # dictionary[seq] = token_representations[j, 1 : len(seq) + 1].mean(0)
    # np.save('biosnap_protein_feature.npy', dictionary)
    # print(len(sequence_representations))

    return df


pro_list_train = df_train['Protein'].unique()
x_train = Get_Protein_Feature(list(pro_list_train))
df_train = pd.merge(df_train, x_train, on='Protein', how='left')
print('train esm is done!\n')

pro_list_val = df_val['Protein'].unique()
x_val = Get_Protein_Feature(list(pro_list_val))
df_val = pd.merge(df_val, x_val, on='Protein', how='left')
print('val esm is done!\n')


pro_list_test = df_test['Protein'].unique()
x = Get_Protein_Feature(list(pro_list_test))
df_test = pd.merge(df_test, x, on='Protein', how='left')
print('test esm is done!\n')


print('ESM feature extraction: pass')
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaModel
#from transformers import TrainingArguments, Trainer, IntervalStrategy

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

from tqdm.auto import tqdm

# Setup
# Load a pretrained transformer model and tokenizer
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "DeepChem/ChemBERTa-77M-MTR"
model_chem = RobertaModel.from_pretrained(model_name, num_labels=2, add_pooling_layer=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_chem = model_chem.to(device)

def get_embeddings(df):
    emblist = []
    #embedding_df = pd.DataFrame(columns=['SMILES'] + [f'chemberta2_feature_{i}' for i in range(1, 385)])

    for index, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):
      # truncate to the maximum length accepted by the model if no max_length is provided
        encodings = tokenizer(row['SMILES'], return_tensors='pt',padding="max_length", max_length=290, truncation=True)
        encodings = encodings.to(device)
        with torch.no_grad():
            output = model_chem(**encodings)
            smiles_embeddings = output.last_hidden_state[0, 0, :]
            #smiles_embeddings = smiles_embeddings.squeeze(0)
            smiles_embeddings = smiles_embeddings.cpu()
            smiles_embeddings = np.array(smiles_embeddings, dtype = np.float64)

            emblist.append(smiles_embeddings)

        # Ensure you move the tensor back to cpu for numpy conversion
        #dic = {**{'SMILES': row['SMILES']}, **dict(zip([f'chemberta2_feature_{i}' for i in range(1, 385)], smiles_embeddings.cpu().numpy().tolist()))}
        #embedding_df.loc[len(embedding_df)] = pd.Series(dic)

    return emblist#smiles_embeddings

df_trainu = df_train.drop_duplicates(subset='SMILES')
df_valu = df_val.drop_duplicates(subset='SMILES')
df_testu = df_test.drop_duplicates(subset='SMILES')

emblist_train = get_embeddings(df_trainu)
df_trainu['fcfp'] = emblist_train

emblist_val = get_embeddings(df_valu)
df_valu['fcfp'] = emblist_val

emblist_test = get_embeddings(df_testu)
df_testu['fcfp'] = emblist_test

# Merge DataFrames on 'SMILES' column
df_train = pd.merge(df_train, df_trainu[['SMILES', 'fcfp']], on='SMILES', how='left')
df_val = pd.merge(df_val, df_valu[['SMILES', 'fcfp']], on='SMILES', how='left')
df_test = pd.merge(df_test, df_testu[['SMILES', 'fcfp']], on='SMILES', how='left')
print('chemBERTa feature extraction: pass\n')
df_train_esm = df_train['esm']
df_val_esm = df_val['esm']
df_test_esm = df_test['esm']

df_train_esm = pd.DataFrame(df_train_esm)
df_val_esm = pd.DataFrame(df_val_esm)
df_test_esm = pd.DataFrame(df_test_esm)

df_train_esm = pd.DataFrame(df_train_esm['esm'].apply(pd.Series))
df_val_esm = pd.DataFrame(df_val_esm['esm'].apply(pd.Series))
df_test_esm = pd.DataFrame(df_test_esm['esm'].apply(pd.Series))

df_train_bert = df_train['fcfp']
df_val_bert = df_val['fcfp']
df_test_bert = df_test['fcfp']

df_train_bert = pd.DataFrame(df_train_bert)
df_val_bert = pd.DataFrame(df_val_bert)
df_test_bert = pd.DataFrame(df_test_bert)

df_train_bert = pd.DataFrame(df_train_bert['fcfp'].apply(pd.Series))
df_val_bert = pd.DataFrame(df_val_bert['fcfp'].apply(pd.Series))
df_test_bert = pd.DataFrame(df_test_bert['fcfp'].apply(pd.Series))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
mol_features_stan_train = scaler.fit_transform(df_train_bert)
protein_features_stan_train = scaler.fit_transform(df_train_esm)

mol_features_stan_val = scaler.fit_transform(df_val_bert)
protein_features_stan_val = scaler.fit_transform(df_val_esm)

mol_features_stan_test = scaler.fit_transform(df_test_bert)
protein_features_stan_test = scaler.fit_transform(df_test_esm)

# Function to replace NaN with the mean of each column
def replace_nan_with_mean_per_column(arr):
    # Iterate over each column
    for i in range(arr.shape[1]):
        col = arr[:, i]
        nan_mask = np.isnan(col)  # Create a mask for NaN values in the column
        if np.any(nan_mask):  # If there are NaN values
            mean_value = np.nanmean(col)  # Calculate the mean of the non-NaN values in the column
            col[nan_mask] = mean_value  # Replace NaN values with the mean of the column
    return arr

# Replace NaN values with the mean of each column
protein_features_stan_train2 = replace_nan_with_mean_per_column(protein_features_stan_train)
protein_features_stan_test2 = replace_nan_with_mean_per_column(protein_features_stan_test)
protein_features_stan_val2 = replace_nan_with_mean_per_column(protein_features_stan_val)

mol_features_stan_test2 = replace_nan_with_mean_per_column(mol_features_stan_test)
mol_features_stan_train2 = replace_nan_with_mean_per_column(mol_features_stan_train)
mol_features_stan_val2 = replace_nan_with_mean_per_column(mol_features_stan_val)

#print("Array after replacing NaN with mean per column:\n", protein_features_stan_test2)


path_train_g = args.train_path#'Data/biosnap/transductive/seed14/train_biosnap14.csv'
path_test_g = args.test_path#'Data/biosnap/transductive/seed14/test_biosnap14.csv'
path_val_g = args.val_path#'Data/biosnap/transductive/seed14/test_biosnap14.csv'





#output_name = 'pdb20trans'
print(run_model(path_train_g, mol_features_stan_train2, protein_features_stan_train2,path_val_g, mol_features_stan_val2, protein_features_stan_train2, path_test_g, mol_features_stan_test2, protein_features_stan_test2 ))
