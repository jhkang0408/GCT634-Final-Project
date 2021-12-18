from PIL import Image
from tqdm import tqdm
from pathlib import Path
import time
import os
import numpy as np
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm.notebook import tqdm 

import torchaudio
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import random_split

import loss_function
import data_utils
import utils
from torch.autograd import Variable


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--datasetPath', type=str, default='./dataset/')
parser.add_argument('--modelPath', type=str, default="./pretrained/I2A(model_148).pt") 
parser.add_argument('--saveDir', type=str, default='./experiment')
parser.add_argument('--gpu', type=str, default='0', help='gpu')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size for training')

args = parser.parse_args()

## basic run command : python train --name temp 
## ex) python test_I2A.py --name '1218(I2A_pretrained)' --datasetPath "C:/Users/User/GCT634 Final/"

if __name__ == '__main__':
    
    #gpu setup.
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    BATCH_SIZE = args.batchSize
    Dataset_Path = args.datasetPath

    #Logging setup.
    save = utils.SaveUtils(args, args.name)

    from model import Image2AudioACVAE, AudioDiscrimitor
    AudioDiscrimitor = AudioDiscrimitor()
    model = Image2AudioACVAE()
    model.load_state_dict(torch.load(args.modelPath))

    _, valid_dataloader, _ = data_utils.get_dataloader(Dataset_Path, BATCH_SIZE)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = model.to(device)
    for  iter, item in enumerate(valid_dataloader):    
    # Move mini-batch to the desired device.
        image, lms, label = item
        image = image.to(device) 
        lms = lms.to(device)
        label = label.to(device)        
        output, _, _, _ = model(image, label)
        
        save.save_image_onlyGT(image, iter)
        save.save_mel(lms.cpu().detach().numpy(), output.cpu().detach().numpy(), iter, label.cpu().detach().numpy())           
        print('save!')
        
        if iter==20:
            break
