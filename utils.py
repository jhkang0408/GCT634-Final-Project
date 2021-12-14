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

from sklearn.manifold import TSNE 

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

#visualize latent space
def save_tsne_img(latent_result,save_label,img_name):
    class_num=13
    save_name="./tsne_result/"+img_name
    #run tsne
    tsne = TSNE(n_components=2, verbose=1, n_iter=300, init='pca',perplexity=5, method='barnes_hut')
    tsne_v = tsne.fit_transform(latent_result)
    #plot
    plt.figure(figsize=(12, 13))  
    instruments = ['bassoon', 'cello', 'clarinet', 'double_bass', 'flute', 'horn', 'oboe', 'sax', 'trombone', 'trumpet', 'tuba', 'viola', 'violin']
    scatter = plt.scatter(tsne_v[:, 0], tsne_v[:, 1],c=save_label, cmap=plt.cm.get_cmap('rainbow', class_num), s=50, label=instruments, alpha=0.95) 
    plt.xlim([-20, 20])      
    plt.ylim([-20, 20])    
    plt.legend(handles=scatter.legend_elements()[0], labels=instruments,loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=7)
    plt.savefig(save_name)
    print(save_name+"  saved image.")

def show_latent_space(iter, with_c, mode, latent, class_pred, label, dataloader,batch_size,latent_result, save_label):
    if iter == 0:
        if with_c:
            latent_result=np.concatenate((latent.cpu().detach().numpy(),class_pred.cpu().detach().numpy()),axis=1) #(16, 141)
        else:
            latent_result=latent.cpu().detach().numpy()
        save_label=label.cpu().detach().numpy()
    else:
        if with_c:
            concat_c=np.concatenate((latent.cpu().detach().numpy(),class_pred.cpu().detach().numpy()),axis=1)
            latent_result=np.concatenate((latent_result, concat_c),axis=0) #(10544, 128) 
        else:
            latent_result=np.concatenate((latent_result,latent.cpu().detach().numpy()),axis=0) #(10544, 128)
        save_label=np.concatenate((save_label,label.cpu().detach().numpy()),axis=0) #(10544,)

        if not iter%((len(dataloader.dataset)//batch_size)-1):#658
            img_name = "["+str(iter)+"] with_c_"+str(with_c)+"_"+mode+".png"
            save_tsne_img(latent_result,save_label,img_name)
            
    return  latent_result, save_label   