import torch
import torch.nn as nn
import torchvision.models as models 
from torchvision import transforms
from torchsummary import summary
import pandas as pd
import random
from glob import glob
import cv2
from tqdm import tqdm

def make_train_test_csv():

    csv_path1 = "C:/Users/hb/Desktop/data/CheXpert-v1.0-small/train.csv"
    csv_path2 = "C:/Users/hb/Desktop/data/CheXpert-v1.0-small/valid.csv"
            
    train_data = pd.read_csv(csv_path1)
    test_data = pd.read_csv(csv_path2)
            
    all_data = pd.concat([train_data, test_data], axis=0)

    length = len(all_data)
    indices = list(range(length))
    random.seed(1996)
    random.shuffle(indices)


    indices_test = indices[:int(0.1 * length)]
    indices_train = indices[int(0.1 * length):]

    all_data = all_data.drop(columns=['Sex','Age','AP/PA'])
    all_data = all_data.drop(columns=['Support Devices','Pleural Effusion','Pleural Other','No Finding'])
    all_data = all_data.fillna(0)
    all_data = all_data.replace(-1,1)

    selected_test = all_data.iloc[indices_test, :]
    selected_train = all_data.iloc[indices_train, :]

    selected_test = selected_test.reset_index()
    selected_train = selected_train.reset_index()

    selected_test = selected_test.drop(columns=['index'])
    selected_train = selected_train.drop(columns=['index'])

    selected_test.to_csv("C:/Users/hb/Desktop/data/CheXpert-v1.0-small/selected_test.csv", index=False)
    selected_train.to_csv("C:/Users/hb/Desktop/data/CheXpert-v1.0-small/selected_train.csv", index=False)
