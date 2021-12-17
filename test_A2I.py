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

    def test(self, dataloader,dir_pth):
        self.model.eval()
        self.model.load_state_dict(torch.load(dir_pth))
        epoch_loss = 0
        loss_NLL_function = nn.CrossEntropyLoss()

        latent_result =[] #np.array([])
        save_label = []
        show_latent=True

        for  iter, item in enumerate(dataloader):
            image, lms, label = item
            image = image.to(self.device) 
            lms = lms.to(self.device)
            label = label.to(self.device)

            output, mean, std, class_pred, latent = self.model(lms, label)

            batch_size = image.shape[0]
            #visualize latent space
            if show_latent:
                with_c=True
                mode = "TEST_A2I"
                latent_result, save_label = show_latent_space(iter, with_c, mode, latent, class_pred, label, dataloader, batch_size,latent_result, save_label)
            # real image
            output_real = self.ImageDiscrimitor(image)
            true_labels = Variable(torch.ones_like(output_real))
            
            loss_G = self.criterion_G(self.ImageDiscrimitor(output), true_labels)
            #recon and latent ELBO
            loss_VAE = loss_function.loss_function(image, output, mean, std)
            #CE the class prediction
            loss_NLL = loss_NLL_function(class_pred, label.detach())

            total_loss = loss_VAE + loss_NLL + 0.0001 * loss_G 

            if iter % 100 == 0:
                log = "[Iter %d] [Train Loss: %.4f] [VAE Loss: %.4f] [Classification Loss: %.4f] [GAN Loss: %.4f]" % (iter, total_loss, loss_VAE, loss_NLL, loss_G)
                print(log)
                save.save_log(log)

            epoch_loss += batch_size * total_loss.item()
        epoch_loss = epoch_loss / len(dataloader.dataset)

        return epoch_loss, output, image


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pth', type=str, default="./weights/model_81.pt", help='directory of pth')
parser.add_argument('--name', type=str, default='./test_result/')
parser.add_argument('--datasetPath', type=str, default='./dataset/') 
parser.add_argument('--saveDir', type=str, default='./experiment')
parser.add_argument('--gpu', type=str, default='0', help='gpu')
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

    test_loss, output_image, gt = runner.test(test_dataloader, args.pth)
    log = "[Test Loss: %.4f]" % (test_loss)
    
    save.save_image(gt, output_image, "test")
    save.save_log(log)
    print(log)

    print("Test Execution time: "+str(time.time()-start))