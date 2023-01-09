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

def resize_NIH():
    all_img_path = glob('C:/Users/hb/Desktop/data/archive/*/*/*.png')

    transform = transforms.Compose([transforms.ToPILImage(), 
                        transforms.Resize(256)])

    for i in tqdm(range(len(all_img_path))):
        img = cv2.imread(all_img_path[i])
        img = transform(img)
        img.save(all_img_path[i],"PNG")