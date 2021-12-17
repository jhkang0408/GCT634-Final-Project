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

class SubURMP(Dataset):
    """Customized SubURMP Dataset.
    datset
        `-- SubURMP
            |-- img
                |-- train
                    |-- bassoon
                        |-- 0001.jpg
                        |-- ...
                        `-- 1735.jpg
                    |-- ...
                    `-- violin
                        |-- 0001.jpg
                        |-- ...
                        `-- 7430.jpg
                `-- test
                    |-- bassoon
                        |-- 0001.jpg
                        |-- ...
                        `-- 0390.jpg
                    |-- ...
                    `-- violin
                        |-- 0001.jpg
                        |-- ...
                        `-- 0945.jpg
            |-- chunk
                |-- train
                    |-- bassoon
                        |-- 0001.wav
                        |-- ...
                        `-- 1735.wav
                    |-- ...
                    `-- violin
                        |-- 0001.wav
                        |-- ...
                        `-- 7430.wav
                `-- test
                    |-- bassoon
                        |-- 0001.wav
                        |-- ...
                        `-- 0390.wav
                    |-- ...
                    `-- violin
                        |-- 0001.wav
                        |-- ...
                        `-- 0945.wav
                # Train
                bassoon:1735
                cello:9800
                clarinet:8125
                double_bass:1270
                flute:5690
                horn:5540
                oboe:4505
                sax:7615
                trombone:8690
                trumpet:1015
                tuba:3285
                viola:6530
                violin:7430 
                # Test
                bassoon:390
                cello:1030
                clarinet:945
                double_bass:1180
                flute:925
                horn:525
                oboe:390
                sax:910
                trombone:805
                trumpet:520
                tuba:525
                viola:485
                violin:945    
    """
 
    def __init__(self, root, train=True):
        super(SubURMP, self).__init__()
        self.root = root
        self.instruments = ['bassoon', 'cello', 'clarinet', 'double_bass', 'flute', 'horn', 'oboe', 'sax', 'trombone', 'trumpet', 'tuba', 'viola', 'violin']
        #self.how_many_train = [1015, 1015, 1015, 1015, 1015, 1015, 1015, 1015, 1015, 1015, 1015, 1015, 1015]
        #self.how_many_test = [390, 390, 390, 390, 390, 390, 390, 390, 390, 390, 390, 390, 390]        
        self.how_many_train = [1735, 9800, 8125, 1270, 5690, 5540, 4505, 7615, 8690, 1015, 3285, 6530, 7430]
        self.how_many_test = [390, 1030, 945, 1180, 925, 525, 390, 910, 805, 520, 525, 485, 945]        
        #self.imgtransform = torchvision.transforms.Compose([torchvision.transforms.Resize((128,128)), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor()])
        self.imgtransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        if train==True:
          dummy_img_paths = [self.root+'Sub-URMP(processed)/IMG/train/'+self.instruments[i]+'/'+str(j+1)+ '.jpg' for i in range(len(self.instruments)) for j in range(self.how_many_train[i])]
          dummy_lms_paths = [self.root+'Sub-URMP(processed)/LMS/train/'+self.instruments[i]+'/'+str(j+1)+ '.npy' for i in range(len(self.instruments)) for j in range(self.how_many_train[i])]
        else:
          dummy_img_paths = [self.root+'Sub-URMP(processed)/IMG/test/'+self.instruments[i]+'/'+str(j+1)+ '.jpg' for i in range(len(self.instruments)) for j in range(self.how_many_test[i])]
          dummy_lms_paths = [self.root+'Sub-URMP(processed)/LMS/test/'+self.instruments[i]+'/'+str(j+1)+ '.npy' for i in range(len(self.instruments)) for j in range(self.how_many_test[i])]
        self.img_paths = dummy_img_paths
        self.lms_paths = dummy_lms_paths
        
        assert isinstance(self.img_paths, list), 'Wrong type. self.paths should be list.'
        if train is True: # 71230, 1015*13
            assert len(self.img_paths) == 71230, 'There are 71,230 train images, but you have gathered %d image paths' % len(self.img_paths)
        else: # 9575, 390*13
            assert len(self.img_paths) == 9575, 'There are 9,575 test images, but you have gathered %d image paths' % len(self.img_paths)
    
    def __getitem__(self, idx):        
        img_path = self.img_paths[idx]
        lms_path = self.lms_paths[idx]
        class_label = (self.instruments).index(img_path.split('/')[len(img_path.split('/'))-2])
        label = torch.tensor(class_label).long()
        
        img = self.imgtransform(Image.open(img_path))
        lms = torch.from_numpy(np.load(lms_path))
        
        return img, lms, label
    
    def __len__(self):
        return len(self.img_paths)
    

def get_dataloader(dataroot, batch_size): 
    train_dataset = SubURMP(dataroot, train=True)    
    train_dataset, valid_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.80), len(train_dataset)-int(len(train_dataset) * 0.80)])
    test_dataset = SubURMP(dataroot, train=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print("# of train_dataset:", len(train_dataset))
    print("# of valid_dataset:", len(valid_dataset))
    print("# of test_dataset: ", len(test_dataset))  
    
    return train_dataloader, valid_dataloader, test_dataloader