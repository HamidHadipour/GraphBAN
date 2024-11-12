import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from prettytable import PrettyTable
from domain_adaptator import ReverseLayerF
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class Trainer(object):
    def __init__(self, model, optim, device,test_dataloader, opt_da=None, discriminator=None,
                 experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
       
        self.test_dataloader = test_dataloader
        #self.teacher_emb = teacher_emb
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        if opt_da:
            self.optim_da = opt_da
        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator
            if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = nn.Linear(in_features=config["DECODER"]["IN_DIM"]*self.n_class, out_features=config["DA"]
                ["RANDOM_DIM"], bias=False).to(self.device)
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False
            elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
                if torch.cuda.is_available():
                    self.random_layer.cuda()
            else:
                self.random_layer = False
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        #self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
  
        

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]
    

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.nb_training) / (
                non_init_epoch * self.nb_training
        )
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact

    def train(self):
        float2str = lambda x: '%0.4f' % x
        
        y_pred  = self.test(dataloader="test")
      
   


        return y_pred

    def save_result(self):
       pass
        
   
        

        
        #test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable_of_prediction_with_trained_model.txt")


        #with open(test_prettytable_file, 'w') as fp:
            #fp.write(self.test_table.get_string())
  

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        test_drug_features, test_protein_features, val_drug_features, val_protein_features = [], [], [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
  
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, sm, v_p,esm, labels) in enumerate(data_loader):
                sm = torch.tensor(sm ,dtype=torch.float32)
                sm = torch.reshape(sm,(sm.shape[0],1,384))
                esm = torch.tensor(esm ,dtype=torch.float32)
                esm = torch.reshape(esm,(sm.shape[0],1,1280))
                v_d, sm,  v_p, esm, labels = v_d.to(self.device),sm.to(self.device), v_p.to(self.device), esm.to(self.device), labels.float().to(self.device)
                device = self.device
                
              
          
                if dataloader == "test":
                    v_d, v_p, f, score = self.model(v_d,sm, v_p,esm, device)
                 
                if self.n_class == 1:
                   
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
              
                
                y_pred = y_pred + n.to("cpu").tolist()
     

        if dataloader == "test":
      
            return y_pred
        else:
            return True
