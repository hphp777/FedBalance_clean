import torch
import torch.nn as nn
import torchvision.models as models 
import argparse
import os, pdb, sys, glob, time
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import warnings
import random

from data.loader.dataloader import NIHTestDataset, ChexpertTestDataset, CIFAR10TestDataset
from data.loader.dataloader import NIHTrainDataset,ChexpertTrainDataset,CIFAR10TrainDatasetALL, CIFAR10TrainDataset
from data.preprocessing.data_loader import NIH_transform, ChexPert_transform
from data.preprocessing.utils import distribute_indices
from model.resnet import resnet56

from methods.Base import client, server

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself ! Cool huh ? :D')
parser.add_argument('--dataset', type = str, default = 'CIFAR10',choices = {'NIH', 'ChexPert','CIFAR10','CIFAR100'}, help = 'This is the path of the training data')
parser.add_argument('--c_round', type = int, default = 20, help = 'The number of communication round')
parser.add_argument('--alpha', type = int, default = 1, help = 'Degree of data heterogeneity')
parser.add_argument('--dir', type = str, default = "C:/Users/hb/Desktop/data/archive")
parser.add_argument('--method', type = str,default ='MOON',  choices = {'FedAvg', 'FedProx', 'MOON', 'FedAlign', 'FedBalance'})
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
clients = []
imbalances = []

def set_weight_client(client_weighting):

    if client_weighting == 'DataAmount':
        for i in range(c_num):
            cw.append(len(clients[i].data) / total_data_num)
    elif client_weighting == 'Imbalance':
        for i in range(c_num):
            cw.append(imbalances[i] / imbalances.sum())
    elif client_weighting == 'Mix':
        temp = []
        for i in range(c_num):
            temp.append((len(clients[i].data) / total_data_num) * (imbalances[i] / imbalances.sum()))
        temp = np.array(temp)
        for i in range(c_num):
            cw.append(temp[i] / temp.sum())

def draw_auc():

    plt.plot(range(len(server_auc)), server_auc)
    plt.savefig('./results/' + args.method + '_auc.png')
    plt.clf()

def draw_acc():

    plt.plot(range(len(server_acc)), server_acc)
    plt.savefig('./results/' + args.method + '_acc.png')
    plt.clf()

# Variance according to the datasets

if args.dataset == "NIH":
    c_num = 5
    model = models.efficientnet_b0(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=14)
    model.to(device)

    data_cnt = 86336
    central_data = NIHTrainDataset(args.data_dir, transform = NIH_transform, indices=list(range(data_cnt)))
    length = len(central_data)
    indices = distribute_indices(length, 1, args.c_num)

    central_server = server(args.dataset, NIHTestDataset(args.data_dir, transform = NIH_transform))
    for i in range(c_num):
        clients.append(client(i, args.method, NIHTrainDataset(i, args.data_dir, transform = NIH_transform, indices=indices[i])))
        imbalances.append(clients[i].imbalance)

elif args.dataset == 'ChexPert':
    c_num = 5
    model = models.efficientnet_b0(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=10)
    model.to(device)

    data_cnt = 86336
    central_data = ChexpertTrainDataset(transform = ChexPert_transform, indices=list(range(data_cnt)))
    length = len(central_data)
    indices = distribute_indices(length, 1, args.c_num)

    central_server = server(args.dataset, ChexpertTestDataset(transform=ChexPert_transform))
    for i in range(c_num):
        clients.append(client(i, args.method, ChexpertTrainDataset(i, transform = ChexPert_transform, indices=indices[i])))
        imbalances.append(clients[i].imbalance)

elif args.dataset == 'CIFAR10':
    c_num = 10
    model = resnet56(10)
    model.to(device)
    central_data = CIFAR10TrainDatasetALL()
    length = len(central_data)

    central_server = server(args.dataset, CIFAR10TestDataset())
    for i in range(c_num):
        clients.append(client(i, args.method, CIFAR10TrainDataset(i)))
        imbalances.append(clients[i].imbalance)

# Variance according to the algorithms

if args.method == 'FedBalance':
    client_weighting = 'Imbalance'
else:
    client_weighting = 'DataAmount'

if __name__ == '__main__':

    weights = [0] * c_num
    weight = model.state_dict()
    server_auc = []
    server_acc = []
    best_acc = 0
    best_auc = 0
    cw = []
    total_data_num = length

    set_weight_client(client_weighting)
    print("Clients' weights : ", cw)

    print("\nCommunication Round 1")

    for i in range(c_num):
        # pool.map(clients[i].train)
        weights[i] = clients[i].train()

    for key in weights[0]:
        weight[key] = sum([weights[i][key] * cw[i] for i in range(c_num)])

    auc ,acc= central_server.test(weight)
    best_acc = acc
    best_auc = auc
    server_auc.append(auc)
    server_acc.append(acc)

    for r in range(2, args.c_round+1):

        print("\nCommunication Round " + str(r))

        for i in range(c_num):
            # pool.map(clients[i].train)
            weights[i] = clients[i].train(updated=True, weight=weight)

        for key in weights[0]:
            weight[key] = sum([weights[i][key] * cw[i] for i in range(c_num)]) 

        torch.save(weight, 'C:/Users/hb/Desktop/code/2.FedBalance/Weight/Finals/FedProx(alpha=0.5)_CIFAR10.pth' )

        # Test
        auc, acc = central_server.test(weight)
        if auc > best_auc:
            best_auc = auc
            
        if acc > best_acc:
            best_acc = acc
        server_auc.append(auc)
        server_acc.append(acc)

        print("Best AUC: ", best_auc)
        print("Best Acc: ", best_acc)


    print("AUCs : ", server_auc)
    print("Best AUC: ", best_auc)
    print("Best Acc: ", best_acc)

    if args.dataset == "NIH" or args.dataset == "ChexPert":
        draw_auc()
    elif args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        draw_acc()





    