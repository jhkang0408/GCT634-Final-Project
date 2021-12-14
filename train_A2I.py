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

class Runner(object):
    def __init__(self, model, ImageDiscrimitor,  lr, sr, save):
        
        

        self.learning_rate = lr
        self.stopping_rate = sr
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.ImageDiscrimitor = ImageDiscrimitor.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer_D = torch.optim.Adam(self.ImageDiscrimitor.parameters(), lr=lr)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        self.scheduler_D = ReduceLROnPlateau(self.optimizer_D, mode='min', factor=0.1, patience=10, verbose=True)

        #GAN loss definition
        self.criterion_D = nn.BCELoss()
        self.criterion_G = nn.BCELoss()

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, epoch, mode='TRAIN'):
        self.model.train() if mode is 'TRAIN' else self.model.eval()

        epoch_loss = 0
        loss_NLL_function = nn.CrossEntropyLoss()
        #pbar = tqdm(dataloader, desc=f'{mode} Epoch {epoch:02}')  # progress bar
        #loop = tqdm(range(len(dataloader)))
        
        #for item in pbar:
        for  iter, item in enumerate(dataloader):
        # Move mini-batch to the desired device.
            image, lms, label = item
            image = image.to(self.device) 
            lms = lms.to(self.device)
            label = label.to(self.device)
            #GT_label = F.one_hot(label, num_classes=13).type(torch.cuda.FloatTensor)                                    
            output, mean, std, class_pred = self.model(lms, label)

            ## update D ##################################################
            for p in self.ImageDiscrimitor.parameters():
                p.requires_grad = True
            self.ImageDiscrimitor.zero_grad()

            # real image
            output_real = self.ImageDiscrimitor(image)
            true_labels = Variable(torch.ones_like(output_real))
            loss_D_real = self.criterion_D(output_real, true_labels)
            #fake image
            fake_image = output.detach()
            D_fake = self.ImageDiscrimitor(fake_image)
            fake_labels = Variable(torch.zeros_like(D_fake))
            loss_D_fake = self.criterion_D(D_fake, fake_labels)
            
            loss_D_total = 0.5 * (loss_D_fake + loss_D_real)
            if mode is 'TRAIN':
                loss_D_total.backward()
                self.optimizer_D.step()

            ## # update G #################################################

            for p in self.ImageDiscrimitor.parameters():
                p.requires_grad = False
            self.ImageDiscrimitor.zero_grad()

            loss_G = self.criterion_G(self.ImageDiscrimitor(output), true_labels)
            #recon and latent ELBO
            loss_VAE = loss_function.loss_function(image, output, mean, std)
            #CE the class prediction
            loss_NLL = loss_NLL_function(class_pred, label.detach())

            total_loss = loss_VAE + loss_NLL + 0.0001 * loss_G
            if mode is 'TRAIN':
                # Perform backward propagation to compute gradients.
                total_loss.backward()
                # Update the parameters.
                self.optimizer.step()
                # Reset the computed gradients.
                self.optimizer.zero_grad()
            
            if iter % 100 == 0:
                #print(GT_label[0])
                log = "[Epoch %d][Iter %d] [Train Loss: %.4f] [VAE Loss: %.4f] [Classification Loss: %.4f] [GAN Loss: %.4f]" % (epoch, iter, total_loss, loss_VAE, loss_NLL, loss_G)
                print(log)
                save.save_log(log)

            batch_size = image.shape[0]
            epoch_loss += batch_size * total_loss.item()
        epoch_loss = epoch_loss / len(dataloader.dataset)
        return epoch_loss, output, image

    def test(self, dataloader):
        epoch_loss = 0
        return epoch_loss

    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        #match net_D lr with generator
        #self.optimizer_D.param_groups[0]['lr'] = self.learning_rate 
        stop = self.learning_rate < self.stopping_rate
        return stop

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--datasetPath', type=str, default='./dataset/') 
parser.add_argument('--saveDir', type=str, default='./experiment')
parser.add_argument('--gpu', type=str, default='0', help='gpu')
parser.add_argument('--numEpoch', type=int, default=10, help='input batch size for training')
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

    from model import Audio2ImageCVAE, ImageDiscrimitor
    ImageDiscrimitor = ImageDiscrimitor()
    model = Audio2ImageCVAE()
    train_dataloader, valid_dataloader, test_dataloader = data_utils.get_dataloader(Dataset_Path, BATCH_SIZE)

    runner = Runner(model=model,ImageDiscrimitor = ImageDiscrimitor , lr = LR, sr = SR, save = save)
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss, _, _ = runner.run(train_dataloader, epoch, 'TRAIN')
        valid_loss, output_image, gt = runner.run(valid_dataloader, epoch, 'VALID')

        log = "[Epoch %d/%d] [Train Loss: %.4f] [Valid Loss: %.4f]" % (epoch + 1, NUM_EPOCHS, train_loss, valid_loss)
        
        save.save_model(model, epoch)
        save.save_image(gt, output_image, epoch)
        save.save_log(log)
        print(log)

        if runner.early_stop(valid_loss, epoch + 1):
            break
    print("Execution time: "+str(time.time()-start))