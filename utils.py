import os
import os.path
import torch
import sys
from torchvision.utils import save_image

import librosa
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
class SaveUtils():
    def __init__(self, args, name):
        self.args = args
        self.save_dir = os.path.join(args.saveDir, name) # make 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.save_dir_image = os.path.join(self.save_dir, 'test_image')
        if not os.path.exists(self.save_dir_image):
            os.makedirs(self.save_dir_image)
        
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')
    
    def save_log(self, log):
        sys.stdout.flush()
        self.logFile.write(log + '\n')
        self.logFile.flush()

    def save_model(self, model, epoch):
            torch.save(model.state_dict(), self.save_dir_model + '/model_'+ str(epoch) +  '.pt')

    def save_image(self, gt, fake, epoch):
        save_image(gt, self.save_dir_image +'/gt_img_'+ str(epoch) +'.png')    
        save_image(fake, self.save_dir_image +'/output_img_'+ str(epoch) +'.png')
    
    def save_mel(self,gt, fake, epoch):
        #print(gt.shape)
        gt = Image.fromarray(np.uint8(gt * 255))
        if gt.mode != 'RGB':
            gt = gt.convert('RGB')
        gt.save(self.save_dir_image +'/gt_mel_'+ str(epoch) +'.png')
        
        fake = Image.fromarray(np.uint8(fake * 255 )) 
        if fake.mode != 'RGB':
            fake = fake.convert('RGB')      
        fake.save(self.save_dir_image +'/output_mel_'+ str(epoch) +'.png')