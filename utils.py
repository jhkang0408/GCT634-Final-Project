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
#import seaborn as sns
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
    
    def save_mel(self, gt, fake, epoch, label):
        cmap = plt.get_cmap('jet') 
        instruments = ['bassoon', 'cello', 'clarinet', 'double_bass', 'flute', 'horn', 'oboe', 'sax', 'trombone', 'trumpet', 'tuba', 'viola', 'violin']
        batch_size = gt.shape[0]
        
        for i in range(batch_size): 
            plt.figure(figsize=(2,4))            
            plt.matshow(gt[i], cmap=cmap)
            plt.clim(-100, 52)
            plt.axis('off')
            plt.title(instruments[label[i]], fontsize=25)
            plt.savefig(self.save_dir_image +'/'+ str(i)+'gt.png')
                   
        gt_images = [Image.open(self.save_dir_image +'/'+ str(x)+'gt.png') for x in range(batch_size)]
        widths, heights = zip(*(i.size for i in gt_images))
        total_width = int(sum(widths)/2) + widths[0]
        max_height = int(max(heights)*2)        
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        i=0
        for im in gt_images:
            if i>=int(batch_size/2)+1:
                new_im.paste(im, (x_offset,im.size[1]))
                x_offset += im.size[0]        
            else:        
                new_im.paste(im, (x_offset,0))
                x_offset += im.size[0]
            if i==int(batch_size/2): 
                x_offset=0
            i+=1
        new_im.save(self.save_dir_image +'/gt_mel_'+ str(epoch) +'.jpg')        
        del gt_images 
        del new_im
        
        for i in range(batch_size):  
            os.remove(self.save_dir_image +'/'+ str(i)+'gt.png')
            
        for i in range(batch_size): 
            plt.figure(figsize=(2,4))
            plt.matshow(fake[i], cmap=cmap)
            plt.clim(-100, 52)
            plt.axis('off')
            plt.title(instruments[label[i]], fontsize=25)
            plt.savefig(self.save_dir_image +'/'+ str(i)+'fake.png')  
            
        fake_images = [Image.open(self.save_dir_image +'/'+ str(x)+'fake.png') for x in range(batch_size)]
        widths, heights = zip(*(i.size for i in fake_images))
        total_width = int(sum(widths)/2) + widths[0]
        max_height = int(max(heights)*2)        
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        i=0
        for im in fake_images:
            if i>=int(batch_size/2)+1:
                new_im.paste(im, (x_offset,im.size[1]))
                x_offset += im.size[0]        
            else:        
                new_im.paste(im, (x_offset,0))
                x_offset += im.size[0]
            if i==int(batch_size/2):
                x_offset=0
            i+=1
        new_im.save(self.save_dir_image +'/output_mel_'+ str(epoch) +'.jpg')           

        del fake_images 
        del new_im 
        
        for i in range(batch_size):  
            os.remove(self.save_dir_image +'/'+ str(i)+'fake.png')        