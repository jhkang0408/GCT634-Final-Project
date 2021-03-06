import os
import os.path
import torch
import sys
import torchaudio
from torchvision.utils import save_image

import librosa
import soundfile as sf

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class SaveUtils():                  
    def __init__(self, args, name):           
        self.args = args
        self.save_dir = os.path.join(args.saveDir, name) # make 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.save_dir_image = os.path.join(self.save_dir, 'IMAGE')
        if not os.path.exists(self.save_dir_image):
            os.makedirs(self.save_dir_image)

        self.save_dir_lms = os.path.join(self.save_dir, 'LMS')
        if not os.path.exists(self.save_dir_lms):
            os.makedirs(self.save_dir_lms) 
            
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

    def save_image_onlyGT(self, gt, epoch):
        save_image(gt, self.save_dir_image +'/gt_img_'+ str(epoch) +'.png')    
    
    def save_mel(self, gt, fake, epoch, label):
        cmap = plt.get_cmap('jet') 
        instruments = ['bassoon', 'cello', 'clarinet', 'double_bass', 'flute', 'horn', 'oboe', 'sax', 'trombone', 'trumpet', 'tuba', 'viola', 'violin']
        batch_size = gt.shape[0]
        
        for i in range(batch_size): 
            plt.figure(figsize=(2,4))            
            plt.matshow(gt[i], cmap=cmap)
            plt.clim(-100, 50)
            plt.axis('off')
            plt.title(instruments[label[i]], fontsize=25)
            plt.savefig(self.save_dir_lms +'/'+ str(i)+'gt.png')
                   
        gt_images = [Image.open(self.save_dir_lms +'/'+ str(x)+'gt.png') for x in range(batch_size)]
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
        new_im.save(self.save_dir_lms +'/gt_mel_'+ str(epoch) +'.jpg')        
        del gt_images 
        del new_im
        
        for i in range(batch_size):  
            os.remove(self.save_dir_lms +'/'+ str(i)+'gt.png')
            
        for i in range(batch_size): 
            plt.figure(figsize=(2,4))
            plt.matshow(fake[i], cmap=cmap)
            plt.clim(-100, 50)
            plt.axis('off')
            plt.title(instruments[label[i]], fontsize=25)
            plt.savefig(self.save_dir_lms +'/'+ str(i)+'fake.png')  
            
        fake_images = [Image.open(self.save_dir_lms +'/'+ str(x)+'fake.png') for x in range(batch_size)]
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
        new_im.save(self.save_dir_lms +'/output_mel_'+ str(epoch) +'.jpg')           

        del fake_images 
        del new_im 
        
        for i in range(batch_size):  
            os.remove(self.save_dir_lms +'/'+ str(i)+'fake.png')        

    def save_mel_onlyGT(self, gt, epoch, label):
        cmap = plt.get_cmap('jet') 
        instruments = ['bassoon', 'cello', 'clarinet', 'double_bass', 'flute', 'horn', 'oboe', 'sax', 'trombone', 'trumpet', 'tuba', 'viola', 'violin']
        batch_size = gt.shape[0]
        
        for i in range(batch_size): 
            plt.figure(figsize=(2,4))            
            plt.matshow(gt[i], cmap=cmap)
            plt.clim(-100, 50)
            plt.axis('off')
            plt.title(instruments[label[i]], fontsize=25)
            plt.savefig(self.save_dir_lms +'/'+ str(i)+'gt.png')
                
        gt_images = [Image.open(self.save_dir_lms +'/'+ str(x)+'gt.png') for x in range(batch_size)]
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
        new_im.save(self.save_dir_lms +'/gt_mel_'+ str(epoch) +'.jpg')        
        del gt_images 
            
    '''
    def save_audio(self, fake, epoch, label):
        instruments = ['bassoon', 'cello', 'clarinet', 'double_bass', 'flute', 'horn', 'oboe', 'sax', 'trombone', 'trumpet', 'tuba', 'viola', 'violin']     
                      
        audio = 10.0**((fake[0]/10))
        audio = librosa.feature.inverse.mel_to_audio(audio, sr=44100, n_fft=2048, hop_length=512)
        print(audio)
        sf.write(self.save_dir_lms +'/'+ instruments[label[0]]+"(output_lms_reconstructed).wav", audio, 44100)            
    '''
    
    