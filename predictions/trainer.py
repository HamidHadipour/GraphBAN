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
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

      
        test_metric_header = ["#", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
      
        self.test_table = PrettyTable(test_metric_header)
        

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]
        self.scaler = StandardScaler()

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
        
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision,cm1,y_pred,thred_optim, cm  = self.test(dataloader="test")
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " f1-score " + str(f1)+ " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))
        test_lst = [str('')] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, thred_optim, test_loss]))

        self.test_table.add_row(test_lst)
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()
        return self.test_metrics, cm, y_pred

    def save_result(self):
   
        

        
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")


        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
  

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
                    #test_drug_features.append(v_d)
                    #test_protein_features.append(v_p)
                if self.n_class == 1:
                    weights = torch.tensor([0.3 if label == 1 else 0.7 for label in labels])
                    weights = weights.to(self.device)
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            if self.experiment:
                self.experiment.log_curve("test_roc curve", fpr, tpr)
                self.experiment.log_curve("test_pr curve", recall, prec)
            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1, cm1,y_pred,thred_optim, cm1 
        else:
            return auroc, auprc, test_loss
