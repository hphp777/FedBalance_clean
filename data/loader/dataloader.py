import glob, os, sys, pdb, time
import torch.nn as nn
import torchvision.models as models 
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from imageio import imread
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import csv
import pandas as pd
import PIL.Image as pilimg
import random
from data.loader import config


class NIHTrainDataset(Dataset):
    def __init__(self,c_num, data_dir, transform = None, indices=None):
        
        self.data_dir = data_dir
        self.transform = transform
        self.df = self.get_df()
        self.make_pkl_dir(config.pkl_dir_path)
        self.the_chosen = indices
        self.new_df = self.train_val_df.iloc[self.the_chosen, :] # this is the sampled train_val data
        self.disease_cnt = [0]*14
        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']
        
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path)):

            self.train_val_df = self.get_train_val_df()
            # pickle dump the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'wb') as handle:
                pickle.dump(self.train_val_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
        else:
            # pickle load the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'rb') as handle:
                self.train_val_df = pickle.load(handle)
        
    
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path)):
            # pickle dump the classes list
            with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'wb') as handle:
                pickle.dump(self.all_classes, handle, protocol = pickle.HIGHEST_PROTOCOL)
                print('\n{}: dumped'.format(config.disease_classes_pkl_path))
        else:
            pass

        for i in range(len(self.new_df)):
            row = self.new_df.iloc[i, :]
            labels = str.split(row['Finding Labels'], '|')
            for lab in labels:  
                lab_idx = self.all_classes.index(lab)
                if lab_idx == 14: # No Finding
                    continue
                self.disease_cnt[lab_idx] += 1

        self.total_ds_cnt = np.array(self.disease_cnt)
        # Normalize the imbalance
        self.imbalance = 0
        difference_cnt = self.total_ds_cnt - self.total_ds_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] * difference_cnt[i]        
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] / difference_cnt.sum()
        # Calculate the level of imbalnce
        difference_cnt -= difference_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = (difference_cnt[i] * difference_cnt[i])
    
        self.imbalance = 1 / difference_cnt.sum()

        # Plot the disease distribution
        plt.figure(figsize=(8,4))
        plt.title('Client{} Disease Distribution'.format(c_num), fontsize=20)
        plt.bar(self.all_classes,self.total_ds_cnt)
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.40)
        plt.xticks(rotation = 90)
        plt.xlabel('Diseases')
        plt.savefig('C:/Users/hb/Desktop/code/3.FedBalance_clean/results/Client{}_disease_distribution.png'.format(c_num))
        plt.clf()

    def get_ds_cnt(self, c_num):

        raw_pos_freq = self.total_ds_cnt
        raw_neg_freq = self.total_ds_cnt.sum() - self.total_ds_cnt

        return raw_pos_freq, raw_neg_freq
            
    def compute_class_freqs(self):
        # total number of patients (rows)
        labels = self.train_val_df ## What is the shape of this???
        N = labels.shape[0]
        positive_frequencies = (labels.sum(axis = 0))/N
        negative_frequencies = 1.0 - positive_frequencies
    
        return positive_frequencies, negative_frequencies

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_train_val_df(self):

        # get the list of train_val data 
        train_val_list = self.get_train_val_list()
        print("train_va_list: ",len(train_val_list))

        train_val_df = pd.DataFrame()
        print('\nbuilding train_val_df...')
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            if filename in train_val_list:
                train_val_df = train_val_df.append(self.df.iloc[i:i+1, :])
        return train_val_df

    def __getitem__(self, index):

        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']
        row = self.new_df.iloc[index, :]
        img = cv2.imread(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        target = torch.zeros(len(self.all_classes))
        new_target = torch.zeros(len(self.all_classes) - 1)
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1            
        if self.transform is not None:
            img = self.transform(img)

        return img, target[:14]
       
    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]

        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df
    
    def get_train_val_list(self):
        f = open("C:/Users/hb/Desktop/data/archive/train_val_list.txt", 'r')
        train_val_list = str.split(f.read(), '\n')
        return train_val_list

    def __len__(self):
        return len(self.the_chosen)
    
    def get_name(self):
        return 'NIH'

    def get_class_cnt(self):
        return 14

class NIHTestDataset(Dataset):

    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        # full dataframe including train_val and test set
        self.df = self.get_df()
        self.make_pkl_dir(config.pkl_dir_path)
        self.disease_cnt = [0]*14
        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']

        # loading the classes list
        with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
            self.all_classes = pickle.load(handle) 
        # get test_df
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.test_df_pkl_path)):
            self.test_df = self.get_test_df()
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'wb') as handle:
                pickle.dump(self.test_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
            print('\n{}: dumped'.format(config.test_df_pkl_path))
        else:
            # pickle load the test_df
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'rb') as handle:
                self.test_df = pickle.load(handle)

        for i in range(len(self.test_df)):
            row = self.test_df.iloc[i, :]
            labels = str.split(row['Finding Labels'], '|')
            for lab in labels:  
                lab_idx = self.all_classes.index(lab)
                if lab_idx == 14: # No Finding
                    continue
                self.disease_cnt[lab_idx] += 1

    def get_ds_cnt(self):
        return self.disease_cnt

    def __getitem__(self, index):
        row = self.test_df.iloc[index, :]
        img = cv2.imread(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        target = torch.zeros(len(self.all_classes)) # 15
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1     
        if self.transform is not None:
            img = self.transform(img)
        return img, target[:14]

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        all_xray_df = pd.read_csv(csv_path)
        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]
        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df

    def get_test_df(self):
        # get the list of test data 
        test_list = self.get_test_list()
        test_df = pd.DataFrame()
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            if filename in test_list:
                test_df = test_df.append(self.df.iloc[i:i+1, :])
        return test_df

    def get_test_list(self):
        f = open( os.path.join('C:/Users/hb/Desktop/data/archive', 'test_list.txt'), 'r')
        test_list = str.split(f.read(), '\n')
        return test_list

    def __len__(self):
        return len(self.test_df)

class ChexpertTrainDataset(Dataset):

    def __init__(self,c_num, transform = None, indices = None):
        
        csv_path = "C:/Users/hb/Desktop/data/CheXpert-v1.0-small/selected_train.csv"
        self.dir = "C:/Users/hb/Desktop/data/"
        self.transform = transform

        self.all_data = pd.read_csv(csv_path)
        self.selecte_data = self.all_data.iloc[indices, :]
        self.class_num = 10
        self.all_classes = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Fracture']
        
        self.total_ds_cnt = np.array(self.disease_cnt)
        # Normalize the imbalance
        self.imbalance = 0
        difference_cnt = self.total_ds_cnt - self.total_ds_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] * difference_cnt[i]        
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] / difference_cnt.sum()
        # Calculate the level of imbalnce
        difference_cnt -= difference_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = (difference_cnt[i] * difference_cnt[i])
    
        self.imbalance = 1 / difference_cnt.sum()

        # Plot the disease distribution
        plt.figure(figsize=(8,4))
        plt.title('Client{} Disease Distribution'.format(c_num), fontsize=20)
        plt.bar(self.all_classes,self.total_ds_cnt)
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.40)
        plt.xticks(rotation = 90)
        plt.xlabel('Diseases')
        plt.savefig('C:/Users/hb/Desktop/code/3.FedBalance_clean/result/Client{}_disease_distribution.png'.format(c_num))
        plt.clf()

    def __getitem__(self, index):

        row = self.selecte_data.iloc[index, :]
        # img = cv2.imread(self.dir + row['Path'])
        img = pilimg.open(self.dir + row['Path'])
        label = torch.FloatTensor(row[2:])
        gray_img = self.transform(img)
        return torch.cat([gray_img,gray_img,gray_img], dim = 0), label

    def __len__(self):
        return len(self.selecte_data)

    def get_ds_cnt(self, c_num):

        raw_pos_freq = self.total_ds_cnt
        raw_neg_freq = self.total_ds_cnt.sum() - self.total_ds_cnt

        return raw_pos_freq, raw_neg_freq

    def get_name(self):
        return 'ChexPert'

    def get_class_cnt(self):
        return 10

class ChexpertTestDataset(Dataset):

    def __init__(self, transform = None):
        
        csv_path = "C:/Users/hb/Desktop/data/CheXpert-v1.0-small/selected_test.csv"
        self.dir = "C:/Users/hb/Desktop/data/"
        self.transform = transform

        self.all_data = pd.read_csv(csv_path)
        self.selecte_data = self.all_data.iloc[:, :]
        # self.selecte_data.to_csv("C:/Users/hb/Desktop/data/CheXpert-v1.0-small/selected_data.csv")
        self.class_num = 10

    def __getitem__(self, index):

        row = self.selecte_data.iloc[index, :]
        img = pilimg.open(self.dir + row['Path'])
        label = torch.FloatTensor(row[2:])
        gray_img = self.transform(img)

        return torch.cat([gray_img,gray_img,gray_img], dim = 0), label

    def get_ds_cnt(self):
        total_ds_cnt = [0] * self.class_num
        for i in range(len(self.selecte_data)):
            row = self.selecte_data.iloc[i, 2:]
            for j in range(len(row)):
                total_ds_cnt[j] += int(row[j])
        return total_ds_cnt

    def __len__(self):
        return len(self.selecte_data)

class CIFAR10TrainDatasetALL(Dataset):

    def __init__(self):
        path = 'C:/Users/hb/Desktop/data/CIFAR10_Client_random/train.csv'
        self.data = pd.read_csv(path)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])

    def __getitem__(self, index):
        
        row = self.data.iloc[index, :]
        img = pilimg.open(row['path'])
        label = torch.FloatTensor(row[2:])
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)

class CIFAR10TrainDataset(Dataset):

    def __init__(self, c_num):
        path = 'C:/Users/hb/Desktop/data/CIFAR10_Client_random/train' + str(c_num) + '.csv'
        self.data = pd.read_csv(path)
        self.class_num = 10
        self.all_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        self.transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])

        self.total_ds_cnt = [0] * self.class_num
        for i in range(len(self.data)):
            row = self.data.iloc[i, 2:]
            for j in range(len(row)):
                self.total_ds_cnt[j] += int(row[j])

        self.total_ds_cnt = np.array(self.total_ds_cnt)
        # Normalize the imbalance
        self.imbalance = 0
        difference_cnt = self.total_ds_cnt - self.total_ds_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] * difference_cnt[i]        
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] / difference_cnt.sum()
        # Calculate the level of imbalnce
        difference_cnt -= difference_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = (difference_cnt[i] * difference_cnt[i])
    
        self.imbalance = 1 / difference_cnt.sum()

        # Plot the disease distribution
        plt.figure(figsize=(8,4))
        plt.title('Client{} Data Distribution'.format(c_num), fontsize=20)
        plt.bar(self.all_classes,self.total_ds_cnt)
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.40)
        plt.xticks(rotation = 90)
        plt.xlabel('Diseases')
        plt.savefig('C:/Users/hb/Desktop/code/3.FedBalance_clean/result/CIFAR10/Client{}_disease_distribution.png'.format(c_num))
        plt.clf()

    def __getitem__(self, index):
        
        row = self.data.iloc[index, :]
        img = pilimg.open(row['path'])
        label = torch.FloatTensor(row[2:])
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)

    def get_ds_cnt(self, c_num):

        raw_pos_freq = self.total_ds_cnt
        raw_neg_freq = self.total_ds_cnt.sum() - self.total_ds_cnt

        return raw_pos_freq, raw_neg_freq

    def get_name(self):
        return 'CIFAR10'

    def get_class_cnt(self):
        return 10

class CIFAR10TestDataset(Dataset):

    def __init__(self):
        path = 'C:/Users/hb/Desktop/data/CIFAR10_Client_random/test.csv'
        self.data = pd.read_csv(path)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])

    def __getitem__(self, index):
        
        row = self.data.iloc[index, :]
        img = pilimg.open(row['path'])
        label = torch.FloatTensor(row[2:])
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)


