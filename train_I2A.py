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
from utils import show_latent_space

class Runner(object):
    def __init__(self, model, lr, sr, save):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        self.learning_rate = lr
        self.stopping_rate = sr
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, epoch, mode='TRAIN'):
        show_latent=False

        self.model.train() if mode is 'TRAIN' else self.model.eval()
        epoch_loss = 0
        loss_NLL_function = nn.CrossEntropyLoss()
        #pbar = tqdm(dataloader, desc=f'{mode} Epoch {epoch:02}')  # progress bar
        #loop = tqdm(range(len(dataloader)))

        latent_result =[] #np.array([])
        save_label = []
        show_latent=False 
        
        #for item in pbar:
        for  iter, item in enumerate(dataloader):
        # Move mini-batch to the desired device.
            image, lms, label = item
            image = image.to(self.device) 
            lms = lms.to(self.device)
            label = label.to(self.device)
   
            output, mean, std, class_pred, latent = self.model(image, label)

            batch_size = image.shape[0]
            #visualize latent space
            if show_latent:
                with_c=True
                mode = "TRAIN_I2A"
                latent_result, save_label = show_latent_space(iter, with_c, mode, latent, class_pred, label, dataloader, batch_size,latent_result, save_label)

            # Compute the loss.            
            loss = loss_function.loss_function(lms, output, mean, std)
            loss_NLL = loss_NLL_function(class_pred, label.detach())
            
            total_loss = loss + loss_NLL
            if iter % 100 == 0:
                log = "[Epoch %d][Iter %d] [Train Loss: %.4f] [VAE Loss: %.4f] [Classification Loss: %.4f]" % (epoch, iter, total_loss, loss, loss_NLL)
                print(log)
                save.save_log(log)

            if mode is 'TRAIN':
                # Perform backward propagation to compute gradients.
                total_loss.backward()
                # Update the parameters.
                self.optimizer.step()
                # Reset the computed gradients.
                self.optimizer.zero_grad()

            batch_size = image.shape[0]
            epoch_loss += batch_size * loss.item()
        epoch_loss = epoch_loss / len(dataloader.dataset)
        return epoch_loss, output, lms, label

    def test(self, dataloader):
        epoch_loss = 0
        return epoch_loss

    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate
        return stop

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--datasetPath', type=str, default='./dataset/')
parser.add_argument('--saveDir', type=str, default='./experiment')
parser.add_argument('--gpu', type=str, default='0', help='gpu')
parser.add_argument('--numEpoch', type=int, default=200, help='input batch size for training')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--sr', type=float, default=1e-6, help='stopping rate')
args = parser.parse_args()

## basic run command : python train --name temp 

if __name__ == '__main__':
    
    #gpu setup.
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    # Training setup.
    LR = args.lr  # learning rate
    SR = args.sr  # stopping rate
    NUM_EPOCHS = args.numEpoch
    BATCH_SIZE = args.batchSize
    Dataset_Path = args.datasetPath
    
    #Logging setup.
    save = utils.SaveUtils(args, args.name)

    from model import Image2AudioCVAE
    model = Image2AudioCVAE()
    train_dataloader, valid_dataloader, test_dataloader = data_utils.get_dataloader(Dataset_Path, BATCH_SIZE)

    runner = Runner(model=model, lr = LR, sr = SR, save = save)
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss, _, _, _ = runner.run(train_dataloader, epoch, 'TRAIN')
        valid_loss, output_image, gt, label = runner.run(valid_dataloader, epoch, 'VALID')

        log = "[Epoch %d/%d] [Train Loss: %.4f] [Valid Loss: %.4f]" % (epoch + 1, NUM_EPOCHS, train_loss, valid_loss)
        
        save.save_model(model, epoch)
        save.save_mel(gt.cpu().detach().numpy(), output_image.cpu().detach().numpy(), epoch, label.cpu().detach().numpy())
        save.save_log(log)
        print(log)

        if runner.early_stop(valid_loss, epoch + 1):
            break
    print("Execution time: "+str(time.time()-start))