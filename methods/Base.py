import torch
import torch.nn as nn
import torchvision.models as models 
import importlib
importlib.reload(models)
import argparse
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import math
import matplotlib.pyplot as plt
import warnings
from cProfile import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, time, random, pdb
import numpy as np
import pandas as pd
import tqdm, pdb
from sklearn.metrics import roc_auc_score,  roc_curve
# import custom dataset classes
from data.loader.dataloader import NIHTrainDataset, NIHTestDataset, ChexpertTrainDataset, ChexpertTestDataset
# import algoruthms
from methods import fedAvg, fedProx, MOON, fedAlign, FedBalance
from model.resnet import resnet56

warnings.filterwarnings(action='ignore')

def q(text = ''): # easy way to exiting the script. useful while debugging
    print('> ', text)
    sys.exit()

class PNB_loss():

    def __init__(self, pos_preq, neg_freq):
        self.beta = 0.9999999
        self.alpha = 10
        self.pos_weights = self.get_inverse_effective_number(self.beta, pos_preq)
        self.neg_weights = self.get_inverse_effective_number(self.beta, neg_freq)
        
        #temp
        self.total = self.pos_weights + self.neg_weights
        self.pos_weights = self.pos_weights / self.total
        self.neg_weights = self.neg_weights / self.total

    def get_inverse_effective_number(self, beta, freq): # beta is same for all classes
        sons = freq / self.alpha # scaling factor
        for i in range(len(freq)):
            sons[i] = math.pow(beta,freq[i])
        sons = np.array(sons)
        En = (1 - sons) / (1 - beta)
        return (1 / En) # the form of vector

    def __call__(self, y_pred, y_true, epsilon=1e-7):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        sigmoid = nn.Sigmoid()
        
        for i in range(len(self.pos_weights)):
            # for each class, add average weighted loss for that class 
            loss_pos =  -1 * torch.mean(self.pos_weights[i] * y_true[:, i] * torch.log(sigmoid(y_pred[:, i]) + epsilon))
            loss_neg =  -1 * torch.mean(self.neg_weights[i] * (1 - y_true[:, i]) * torch.log(1 -sigmoid( y_pred[:, i]) + epsilon))
            loss += self.pos_weights[i] * (loss_pos + loss_neg)
            # loss = (1 / self.neg_weights[i]) * loss * 0.05
        return loss

class server():

    def __init__(self, dataset_name, dataset):

        self.num_workers = 4
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 32
        self.lr = 0.01
        self.d_name = dataset_name
        self.test_loader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = not True)
        self.width_range = [0.25, 1]

        if self.d_name == 'NIH':
            # EfficientNetB0
            self.model = models.efficientnet_b0(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=14)
            self.model.to(self.device)
            self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        elif self.d_name == 'ChexPert':
            self.model = models.efficientnet_b0(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=10)
            self.model.to(self.device)
            self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        elif self.d_name == 'CIFAR10':
            # ResNet
            self.model = resnet56(10)
            self.model.to(self.device)
            self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
            self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        elif self.d_name == 'CIFAR100':
            # ResNet
            self.model = resnet56(100)
            self.model.to(self.device)
            self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
            self.loss_fn = nn.CrossEntropyLoss().to(self.device)     

        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)            

        print('\n-----Initial Dataset Information(Server)-----')
        print('num images in test dataset: {}'.format(len(dataset)))
        print('-------------------------------------')

    def test(self,weight):

        k=0
        total = 0
        correct = 0

        self.model.load_state_dict(weight)
        self.model.eval()

        running_val_loss = 0
        val_loss_list = []
        val_loader_examples_num = len(self.test_loader)
        sigmoid = torch.nn.Sigmoid()

        if self.d_name == "NIH":
            class_cnt = 14
            probs = np.zeros((val_loader_examples_num, class_cnt), dtype = np.float32)
            gt    = np.zeros((val_loader_examples_num, class_cnt), dtype = np.float32)
        elif self.d_name == "ChexPert":
            class_cnt = 10
            probs = np.zeros((val_loader_examples_num, class_cnt), dtype = np.float32)
            gt    = np.zeros((val_loader_examples_num, class_cnt), dtype = np.float32)


        with torch.no_grad():  
            for batch_idx, (img, target) in enumerate(self.test_loader):
            
                per = ((batch_idx+1)/len(self.test_loader))*100
                a_, b_ = divmod(per, 1)
                print(f'{str(batch_idx+1).zfill(len(str(len(self.test_loader))))}/{str(len(self.test_loader)).zfill(len(str(len(self.test_loader))))} ({str(int(a_)).zfill(2)}.{str(int(100*b_)).zfill(2)} %)', end = '\r')
        
                img = img.to(self.device)
                target = target.to(self.device)    
        
                
                if self.d_name == 'NIH' or self.d_name == "ChexPert":
                    out = self.model(img)       
                    loss = self.loss_fn(out, target)    
                    running_val_loss += loss.item()*img.shape[0]
                    val_loss_list.append(loss.cpu().detach().numpy())
                    preds = np.round(sigmoid(out).cpu().detach().numpy())
                    targets = target.cpu().detach().numpy()
                    total += len(targets)*class_cnt
                    correct += (preds == targets).sum()
                    probs[k: k + out.shape[0], :] = out.cpu()
                    gt[   k: k + out.shape[0], :] = target.cpu()
                    k += out.shape[0]

                elif self.d_name == 'CIFAR10' or self.d_name == "CIFAR100":
                    out = torch.log_softmax(self.model(img), dim=1)        
                    target = torch.argmax(target, dim=1)
                    loss = self.loss_fn(out, target)    
                    prediction = out.max(1, keepdim=True)[1] # index
                    preds = prediction.squeeze().cpu().detach().numpy()
                    targets = target.cpu().detach().numpy()
                    total += len(targets)
                    correct += (preds == targets).sum()

            accuracy = correct/total
            print("Test Accuracy: ", correct/total)
            try:
                roc_auc = roc_auc_score(gt, probs)
            except:
                roc_auc = 0

        return roc_auc, accuracy


class client():

    def __init__(self, c_num = None, method = None, data = None):

        # define the learning rate
        self.c_num = c_num
        self.method = method
        self.data = data
        self.d_name = self.data.get_name()
        self.lr = 0.01
        self.num_workers = 4
        self.class_cnt = 10
        self.batch_size = 32
        self.local_epoch = 10
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.imbalance = data.imbalance
        self.width_range = [0.25, 1]

        if self.method == "MOON":
            if self.d_name == 'NIH':
                # EfficientNetB0
                self.model = models.efficientnet_b0(pretrained=True)
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=14)
                self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
            elif self.d_name == 'ChexPert':
                self.model = models.efficientnet_b0(pretrained=True)
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=10)
                self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
            elif self.d_name == 'CIFAR10':
                # ResNet
                self.model = resnet56(10)
                self.model.to(self.device)
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
                self.prev_model = resnet56(10)
                self.prev_model.to(self.device)
                self.prev_model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
                self.glob_model = resnet56(10)
                self.glob_model.to(self.device)
                self.glob_model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
                self.loss_fn = nn.CrossEntropyLoss().to(self.device)
            elif self.d_name == 'CIFAR100':
                # ResNet
                self.model = models.resnet50(pretrained=True)
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, 100) 
                self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        else:
            if self.d_name == 'NIH':
                self.model = models.efficientnet_b0(pretrained=True)
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=14)
                self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
            elif self.d_name == 'ChexPert':
                self.model = models.efficientnet_b0(pretrained=True)
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=10)
                self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
            elif self.d_name == 'CIFAR10':
                self.model = resnet56(10)
                self.model.to(self.device)
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
                self.loss_fn = nn.CrossEntropyLoss().to(self.device)
            elif self.d_name == 'CIFAR100':
                self.model = resnet56(10)
                self.model.to(self.device)
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
                self.loss_fn = nn.CrossEntropyLoss().to(self.device)

        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)

        # define the loss function
        if self.method == 'FedBalance':
            pos_freq, neg_freq = self.data.get_ds_cnt(self.c_num)
            self.loss_fn = PNB_loss(pos_freq, neg_freq)
            self.fit = FedBalance.fit
        elif self.method == 'FedAvg':
            self.fit = fedAvg.fit
        elif self.method == 'FedProx':
            self.fit = fedProx.fit
        elif self.method == 'MOON':
            self.fit = MOON.fit
        elif self.method == 'FedAlign':
            self.fit = fedAlign.fit    

        print('\n-----Initial Dataset Information({})-----'.format(self.c_num))
        print('num images in dataset   : {}'.format(len(self.data)))
        print('-------------------------------------')

    def count_parameters(self, model): 
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num_parameters/1e6 # in terms of millions

    def q(text = ''): # easy way to exiting the script. useful while debugging
        print('> ', text)
        sys.exit()

    def train(self, updated = False, weight = None):
        
        print("\nClient" + str(self.c_num) + " Staging==============================================")
        
        if updated == True:
            self.model.load_state_dict(weight)
        for name, param in self.model.named_parameters(): # all requires_grad by default, are True initially
            param.requires_grad = True 

        # making empty lists to collect all the losses
        losses_dict = {'epoch_train_loss': [], 'epoch_val_loss': [], 'total_train_loss_list': [], 'total_val_loss_list': []}

        # checking the layers which are going to be trained (irrespective of args.resume)
        trainable_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                layer_name = str.split(name, '.')[0]
                if layer_name not in trainable_layers: 
                    trainable_layers.append(layer_name)
        print('following are the trainable layers...')
        print(trainable_layers)
        
        if self.method == 'MOON':
            if updated == True:
                self.glob_model.load_state_dict(weight)
            weight = self.fit(self.data, self.model, self.prev_model, self.glob_model, 
                                 self.optimizer, self.loss_fn, losses_dict,
                                final_epochs = self.local_epoch,
                                bs = self.batch_size)
            self.prev_model.load_state_dict(weight)
        else:
            weight = self.fit(self.data, self.model, self.loss_fn, 
                                        self.optimizer, losses_dict,
                                        final_epochs = self.local_epoch,
                                        bs = self.batch_size)

        return weight