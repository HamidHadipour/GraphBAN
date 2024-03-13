import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch.nn import Linear, Dropout
import torch.optim as optim
from torchmetrics import AUROC
import random
import torch_geometric.transforms as T
from torch import nn

# Import other necessary libraries here...

def set_seeds(seed_value=10):
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

def load_node_csv(path, index_col, encoders=None, **kwargs):

    df = pd.read_csv(path, index_col=index_col, **kwargs)
    #If the dataset is parquet uncomment below line and comment above line
    #df = pd.read_parquet(path, index_col=index_col, **kwargs)
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
        self.dropout = Dropout(p=0.5)
    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index).relu()
        #x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        #x = self.dropout(x)
        x = self.conv3(x, edge_index).relu()
        #x = self.dropout(x)
        #x = self.conv4(x, edge_index).relu()

        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        #self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin1 = Linear(2*hidden_channels,1)
        self.lin2 = Linear(3*hidden_channels, 2*hidden_channels)
        self.lin3 = Linear(2*hidden_channels, hidden_channels)
        self.lin4 = Linear(hidden_channels, 1)
        self.dropout = Dropout(p=0.5)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        x = torch.cat([z_dict['chem'][row], z_dict['protein'][col]], dim=-1)
        #c = z_dict['chem'][row]
        #g = self.lin1(c)
        #x1 = self.dropout(x)
        #z1 = self.dropout(x)
        z1 = self.lin1(x)#.relu()


        #z2 = self.lin2(z)
        #z2 = self.lin2(z).relu()
        #z3 = self.lin3(z2)
        #z4 = self.lin4(z3)
        #z3 = torch.sigmoid(z2)
        z5 = z1.view(-1)

        return z5,x,edge_label_index


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
    return  float(loss)

@torch.no_grad()
def test_auroc(model, data):
    model.eval()
    pred,emb,ed = model(data.x_dict, data.edge_index_dict,
                 data['chem', 'CPI', 'protein'].edge_label_index)
    target = data['chem', 'CPI', 'protein'].edge_label.float()
    auroc = AUROC(task = 'binary')
    aurocscore = auroc(pred, target)
    return float(aurocscore), emb,ed

def run_model(epochs, output_file_path, data_file_path):
 
  set_seeds()
  cp_path = data_file_path
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
  data['chem'].num_nodes =len(cp_mapping_train)  # Users do not have any features.
  data['protein'].num_nodes =len(cp_mapping_p_train)
  data['chem', 'CPI', 'protein'].edge_index = edge_index_train.type(torch.int64)
  data['chem'].x = torch.eye(data['chem'].num_nodes,128)
  del data['chem'].num_nodes
  data['protein'].x = torch.eye(data['protein'].num_nodes,128)
  del data['protein'].num_nodes
  data['chem', 'CPI', 'protein'].edge_label = edge_label_train
  data['chem', 'CPI', 'protein'].edge_label_index = edge_label_index_train
  data = T.ToUndirected()(data)
  del data['protein', 'rev_CPI', 'chem'].edge_label
  #print(data)
  #print(data['chem', 'CPI', 'protein'].edge_label_index)


  

    # Initialize and prepare the model, data, etc.
  hidden_channels=128
  model = Model(data, hidden_channels=hidden_channels)
  with torch.no_grad():
    model.encoder2(data.x_dict, data.edge_index_dict)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(1, epochs + 1):
    
    loss = train(model, optimizer, data)
    train_rmse, emb, ed = test_auroc(model, data)
     
  emb_np = emb.detach().cpu().numpy()
  emb_df = pd.DataFrame(emb_np)
  embcol = []
  for i in range(hidden_channels*2):
    embcol.append(str(i))

  emb_df = pd.DataFrame(data = emb_df, columns=embcol)
  emb_df.to_parquet(output_file_path)
  message = f"Saved embeddings to {output_file_path}"
  return message
  #print(f"Saved embeddings to {output_file_path}")