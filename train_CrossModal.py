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

        #Cosine Similarity 
        self.criterion_Cosim = nn.CosineSimilarity(dim=1, eps=1e-6)
        
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
            out_image, latent_a, mean_a, logvar_a, class_pred_a, out_lms, latent_v, mean_v, logvar_v, class_pred_v = self.model(image, lms, label)

            ## update D ##################################################
            for p in self.ImageDiscrimitor.parameters():
                p.requires_grad = True
            self.ImageDiscrimitor.zero_grad()

            # real image
            output_real = self.ImageDiscrimitor(image)
            true_labels = Variable(torch.ones_like(output_real))
            loss_D_real = self.criterion_D(output_real, true_labels)
            #fake image
            fake_image = out_image.detach()
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

            # Audio -> Image
            loss_G = self.criterion_G(self.ImageDiscrimitor(out_image), true_labels)
            loss_VAE_image = loss_function.loss_function(image, out_image, mean_a, logvar_a)
            loss_NLL_Audio = loss_NLL_function(class_pred_a, label.detach())
            
            # Image -> Audio
            loss_VAE_Audio = loss_function.loss_function(lms, out_lms, mean_v, logvar_v)
            loss_NLL_Image = loss_NLL_function(class_pred_v, label.detach())
            
            # Cosine Similarity            
            loss_Cosim = 0.1*(1-self.criterion_Cosim(torch.cat([mean_a, logvar_a],1), torch.cat([mean_v, logvar_v],1)).mean())
            total_loss = (loss_VAE_image + loss_NLL_Audio + 0.0001*loss_G) + (loss_VAE_Audio + loss_NLL_Image) + loss_Cosim
            
            if mode is 'TRAIN':
                # Perform backward propagation to compute gradients.
                total_loss.backward()
                # Update the parameters.
                self.optimizer.step()
                # Reset the computed gradients.
                self.optimizer.zero_grad()
            
            if iter % 100 == 0:
                log = "[Epoch %d][Iter %d] [Total Loss: %.4f] [Cosine Similarity Loss: %.4f] \n Image2Audio: [VAE Loss: %.4f] [Classification Loss: %.4f]" % (epoch, iter, total_loss, loss_Cosim, loss_VAE_Audio, loss_NLL_Image )
                print(log)
                save.save_log(log)
                #log = "[Epoch %d][Iter %d] [Train Loss: %.4f] [VAE Image Loss: %.4f] [VAE Audio Loss: %.4f] [Cosine Similarity Loss: %.4f]" % (epoch, iter, total_loss, loss_VAE_image, loss_VAE_Audio, loss_Cosim)
                log = "Audio2Image: [VAE Loss: %.4f] [Classification Loss: %.4f] [GAN Loss: %.4f]" % (loss_VAE_image, loss_NLL_Audio, loss_G)
                print(log)
                save.save_log(log)

            batch_size = image.shape[0]
            epoch_loss += batch_size * total_loss.item()
        epoch_loss = epoch_loss / len(dataloader.dataset)
        return epoch_loss, out_image, image, out_lms, lms, label

    def test(self, dataloader):
        epoch_loss = 0
        return epoch_loss

    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        #match net_D lr with generator
        self.optimizer_D.param_groups[0]['lr'] = self.learning_rate 
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

    from model import CrossModal, ImageDiscrimitor
    ImageDiscrimitor = ImageDiscrimitor()
    model = CrossModal()
    train_dataloader, valid_dataloader, test_dataloader = data_utils.get_dataloader(Dataset_Path, BATCH_SIZE)

    runner = Runner(model=model,ImageDiscrimitor = ImageDiscrimitor , lr = LR, sr = SR, save = save)
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss, _, _, _, _, _ = runner.run(train_dataloader, epoch, 'TRAIN')
        valid_loss, out_image, image, out_lms, lms, label = runner.run(valid_dataloader, epoch, 'VALID')
        
        log = "[Epoch %d/%d] [Train Loss: %.4f] [Valid Loss: %.4f]" % (epoch + 1, NUM_EPOCHS, train_loss, valid_loss)
        
        save.save_model(model, epoch)
        save.save_image(image, out_image, epoch)
        save.save_mel(lms.cpu().detach().numpy(), out_lms.cpu().detach().numpy(), epoch, label.cpu().detach().numpy())
        save.save_log(log)
        print(log)

        if runner.early_stop(valid_loss, epoch + 1):
            break
    print("Execution time: "+str(time.time()-start))