import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from tqdm import tqdm
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')

try:
  from rdkit import Chem
  from rdkit.Chem.Draw import IPythonConsole
except ImportError:
  print('Stopping RUNTIME. Colaboratory will restart automatically. Please run again.')
  exit()

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss
'''
def binary_cross_entropy(pred_output, labels, weights):
    loss_fct = torch.nn.BCELoss(weight=weights)
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels.float())
    return n, loss
'''
def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent

class GraphBAN(nn.Module):
    def __init__(self, **config):
        super(GraphBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.molecule_FCFP = LinearTransform()
        self.protein_esm = LinearTransform_esm()
        self.mol_fusion = molFusion()
        self.pro_fusion = proFusion()
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
        self.scaler = StandardScaler()

    def forward(self, bg_d, bg_smiles, v_p,v_p_esm, device, mode="train"):
        v_d = self.drug_extractor(bg_d)

        v_smiles_fcfp = self.molecule_FCFP(bg_smiles)

        v_fusion = self.mol_fusion(v_d, v_smiles_fcfp)


        v_p = self.protein_extractor(v_p)
        v_p_esm = self.protein_esm(v_p_esm)
        v_p_fusion = self.pro_fusion(v_p, v_p_esm)
        f, att = self.bcn(v_fusion, v_p_fusion)
        
        #f_numpy = f.detach().cpu().numpy()
        #_scaled = torch.tensor(self.scaler.fit_transform(f_numpy))
        #f_scaled = f_scaled.to(device)
        #f = torch.cat((v_fusion ,v_p), dim=0)
        
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_fusion, v_p, f, score
        elif mode == "eval":
            return v_fusion, v_p, score, f



class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class LinearTransform(nn.Module):
    def __init__(self):
        super(LinearTransform, self).__init__()
        self.linear1 = nn.Linear(384, 512)#for seed 12 for better score it was on 384>64>128
        self.linear2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        # Reshape the input tensor to [batch_size, 1024]
        x = x.view(x.size(0), -1)

        # Apply the first linear layer
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = x.unsqueeze(1)  # Add a singleton dimension to match [batch_size, 1, 512]

        # Apply the second linear layer
        x = torch.relu(self.linear2(x))

        return x

class LinearTransform_esm(nn.Module):
    def __init__(self):
        super(LinearTransform_esm, self).__init__()
        self.linear1 = nn.Linear(1280, 512)
        self.linear2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        # Reshape the input tensor to [batch_size, 1024]
        x = x.view(x.size(0), -1)

        # Apply the first linear layer
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = x.unsqueeze(1)  # Add a singleton dimension to match [batch_size, 1, 512]

        # Apply the second linear layer
        x = torch.relu(self.linear2(x))

        return x

class molFusion(nn.Module):
  def __init__(self):
    super(molFusion, self).__init__()

  def forward(self, A, B):

  # 1. Perform element-wise multiplication A * B
    result_1 = torch.matmul(A, B.transpose(1,2))

# 2. Perform element-wise multiplication (result_1) * (transpose of B)
    result_2 = torch.matmul(result_1, B)

# 3. Perform element-wise addition with A
    final_result = torch.add(result_2, A)
    return final_result



class proFusion(nn.Module):
  def __init__(self):
    super(proFusion, self).__init__()

  def forward(self, A, B):

  # 1. Perform element-wise multiplication A * B
    result_1 = torch.matmul(A, B.transpose(1,2))

# 2. Perform element-wise multiplication (result_1) * (transpose of B)
    result_2 = torch.matmul(result_1, B)

# 3. Perform element-wise addition with A
    final_result = torch.add(result_2, A)
    return final_result
    
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
