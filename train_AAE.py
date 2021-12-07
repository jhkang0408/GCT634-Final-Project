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
from torch.autograd import Variable

import data_utils
import utils

class Runner(object):
    def __init__(self, encoder, decoder, discriminator, lr, sr):        
        self.learning_rate = lr
        self.stopping_rate = sr
        self.EPS = 1e-15        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
        # encoder, decoder, discriminator
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.discriminator = discriminator.to(self.device)
        # encoder/decoder optimizers
        self.optim_encoder = torch.optim.Adam(encoder.parameters(), lr=lr) 
        self.optim_decoder = torch.optim.Adam(decoder.parameters(), lr=lr)  
        # regularizer optimizers
        self.optim_encoder_gen = torch.optim.Adam(encoder.parameters(), lr=lr/2)
        self.optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr/2) 
        
        #self.scheduler = ReduceLROnPlateau(self.optim_encoder, mode='min', factor=0.1, patience=10, verbose=True)
             
        
    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, epoch, mode='TRAIN'):
        if mode=='TRAIN':
            self.encoder.train()        
            self.decoder.train()
            self.discriminator.train()
        else :
            self.encoder.eval()
            self.decoder.eval()
            self.discriminator.eval()    

        epoch_loss = 0
        loss_NLL_function = nn.CrossEntropyLoss()
        
        #for item in pbar:
        for  iter, item in enumerate(dataloader):
        # Move mini-batch to the desired device.
            image, lms, label = item
            
            image = image.to(self.device) 
            lms = lms.to(self.device)
            label = label.to(self.device)
            
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            self.discriminator.zero_grad()   
            
            # Reconstruction
            #################################################################################
            audio_latent, class_pred = self.encoder(lms) #, label
            output = self.decoder(audio_latent, class_pred)
            
            loss_reconstruction = F.mse_loss(output, image) 
            loss_classification = loss_NLL_function(class_pred, label.detach())
            
            if mode is 'TRAIN':
                (loss_reconstruction + loss_classification).backward()
                self.optim_decoder.step()
                self.optim_encoder.step()
            #################################################################################
            
            # Discrimination
            #################################################################################
            self.encoder.eval()
            audio_latent_real_gauss = Variable(torch.randn(image.size()[0], 128) * 1.).cuda()
            real_gauss = self.discriminator(audio_latent_real_gauss)            
            audio_latent_fake_gauss, _ = self.encoder(lms)#, label
            fake_gauss = self.discriminator(audio_latent_fake_gauss)
            
            loss_discrimination = 0.1 * -torch.mean(torch.log(real_gauss + self.EPS) + torch.log(1 - fake_gauss + self.EPS))
            
            if mode is 'TRAIN': 
                loss_discrimination.backward() 
                self.optim_discriminator.step()                            
            #################################################################################
            
            # Generation
            ################################################################################# 
            self.encoder.train()
            audio_latent_fake_gauss, _ = self.encoder(lms) #,label
            fake_gauss = self.discriminator(audio_latent_fake_gauss)
            
            loss_generation = 0.1 * -torch.mean(torch.log(fake_gauss + self.EPS))
            if mode is 'TRAIN': 
                loss_generation.backward()
                self.optim_encoder_gen.step()
            #################################################################################  
                                   
            total_loss = loss_reconstruction + loss_classification + loss_discrimination + loss_generation            
            
            if iter % 100 == 0:
                print("[Epoch %d][Iter %d] [Train Loss: %.4f] [Reconstruction Loss: %.4f] [Classification Loss: %.4f]" % (epoch, iter, total_loss, loss_reconstruction, loss_classification))                                  
            
            batch_size = image.shape[0]
            epoch_loss += batch_size * total_loss.item()
            
        epoch_loss = epoch_loss / len(dataloader.dataset)
        return epoch_loss, output, image

    def test(self, dataloader):
        epoch_loss = 0
        return epoch_loss
    
    '''
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate
        return stop
    '''
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
    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    # Training setup.
    LR = args.lr  # learning rate
    SR = args.sr  # stopping rate
    NUM_EPOCHS = args.numEpoch
    BATCH_SIZE = args.batchSize
    Dataset_Path = args.datasetPath

    #Logging setup.
    save = utils.SaveUtils(args, args.name)

    from model_AAE import AudioEncoder
    from model_AAE import AudNet
    from model_AAE import ImageDecoder
    from model_AAE import Discriminator
    #encoder = AudioEncoder()
    encoder = AudNet(norm="bn")
    decoder = ImageDecoder()
    discriminator = Discriminator()
    
    train_dataloader, valid_dataloader, test_dataloader = data_utils.get_dataloader(Dataset_Path, BATCH_SIZE)

    runner = Runner(encoder=encoder, decoder=decoder, discriminator=discriminator, lr = LR, sr = SR)
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss, _, _ = runner.run(train_dataloader, epoch, 'TRAIN')
        valid_loss, output_image, gt = runner.run(valid_dataloader, epoch, 'VALID')

        log = "[Epoch %d/%d] [Train Loss: %.4f] [Valid Loss: %.4f]" % (epoch + 1, NUM_EPOCHS, train_loss, valid_loss)
        
        #save.save_model(model, epoch)
        save.save_image(gt, output_image, epoch)
        save.save_log(log)
        print(log)
        '''
        if runner.early_stop(valid_loss, epoch + 1):
            break
        '''
    print("Execution time: "+str(time.time()-start))