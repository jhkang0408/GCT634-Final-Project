import torchaudio

import torch
import torch.nn as nn
import torch.nn.functional as F


class Audio_block(nn.Module):
      def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        super(Audio_block, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=pooling, stride=pooling)
    
      def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class AudioEncoder(nn.Module):
    '''
    input: audio wave
    output:  latent z torch.Size([1, 128])
    '''
    def __init__(self):
        super(AudioEncoder, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=2048, hop_length=512, f_min=0.0, f_max=8000.0, n_mels=128)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Audio_block(input_channels = 1, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=4)
        self.layer2 = Audio_block(input_channels = 64, output_channels = 64 * 2, kernel_size=3, stride=1, padding=1, pooling=3)
        self.layer3 = Audio_block(input_channels = 64 * 2, output_channels = 64 * 2, kernel_size=3, stride=1, padding=1, pooling=3)

        self.final_pool = nn.AdaptiveAvgPool2d(1)   
        self.linear_mean = nn.Linear(128+13, 128)
        self.linear_std = nn.Linear(128+13, 128)
        
    def forward(self, x , c):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x.unsqueeze(1))                        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_pool(x)
        c = (F.one_hot(c, num_classes=13)).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat([x, c], 1)#add class information
        ## disentagle
        return self.linear_mean(x.squeeze(-1).squeeze(-1)), self.linear_std(x.squeeze(-1).squeeze(-1))


class decoder_block(nn.Module):
    def __init__(self, InChannel, OutChannel):
        super(decoder_block, self).__init__()
        self.ConvTrans = nn.ConvTranspose2d(InChannel, OutChannel, 4, 2, 1, bias=False)
        self.BN = nn.BatchNorm2d(OutChannel)
        self.ReLU = nn.ReLU(True)
            
    def forward(self, x):
        x = self.ConvTrans(x)
        x = self.BN(x)
        x = self.ReLU(x)
        return x

class ImageDecoder(nn.Module):
    def __init__(self):
        '''
        input: latent z torch.Size([1, 128])
        output:  RGB image / torch.Size([3, 256, 256])
        '''
        super(ImageDecoder, self).__init__()
        self.de_block1 = decoder_block(128+13 , 64)
        self.de_block2 = decoder_block(64 , 32)
        
        # do upscaling
        self.Upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   
        self.de_block3 = decoder_block(32 , 16)
        self.de_block4 = decoder_block(16 , 8)
     
        # do upscaling
        self.Upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.de_block5 = decoder_block(8 , 4)
        self.de_block6 = decoder_block(4 , 3)
        self.last_conv = nn.Conv2d(3, 3, kernel_size= 3, padding= 1)

            
    def forward(self, x, c):
        ## disentagle
        c = F.one_hot(c, num_classes=13)
        x = torch.cat([x, c], 1)#add class information
        x = self.de_block1(x.unsqueeze(-1).unsqueeze(-1)) # torch.Size([1, 64, 2, 2])
        x = self.de_block2(x) # torch.Size([1, 32, 4, 4])
        x = self.Upsample1(x) # torch.Size([1, 32, 8, 8])
        x = self.de_block3(x) # torch.Size([1, 16, 16, 16])
        x = self.de_block4(x) # torch.Size([1, 8, 32, 32])
        x = self.Upsample2(x) # torch.Size([1, 8, 64, 64])
        x = self.de_block5(x) # torch.Size([1, 4, 128, 128])
        x = self.de_block6(x) # torch.Size([1, 3, 256, 256])

        x = self.last_conv(x)
        return x


class Audio2ImageCVAE(nn.Module):
    def __init__(self):
        super(Audio2ImageCVAE, self).__init__()
        '''
        input: audio wave
        output:  RGB image / torch.Size([3, 256, 256])
        '''
        self.AudioEncoder = AudioEncoder()
        self.ImageDecoder = ImageDecoder()

    def sampling(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, x, c):
        mean, logvar = self.AudioEncoder(x, c)
        latent = self.sampling(mean, logvar)
        out = self.ImageDecoder(latent, c)   
        return out, mean, logvar # torch.Size([3, 256, 256])
    
    
    
    
    
    