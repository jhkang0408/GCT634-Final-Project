import torchaudio

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classification_Branch(nn.Module):
    def __init__(self, input_dim = 128, output_class = 13):
        super(Classification_Branch, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2 )
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_dim // 2, input_dim // 4 )
        self.relu2 = nn.ReLU()
        self.linear_last = nn.Linear(input_dim // 4, output_class)
        #self.LogSoftMax = nn.LogSoftmax()
      
    def forward(self, x):
        #print(": ", x.shape)
        x = self.relu1(self.linear1(x.squeeze(-1).squeeze(-1)))
        x = self.relu2(self.linear2(x))
        #x = self.LogSoftMax(self.linear_last(x))
        x = self.linear_last(x)
        return x



class Con2d_block(nn.Module):
      def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        super(Con2d_block, self).__init__()
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
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Con2d_block(input_channels = 1, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer2 = Con2d_block(input_channels = 64, output_channels = 64 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer3 = Con2d_block(input_channels = 64 * 2, output_channels = 64 * 2 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer4 = Con2d_block(input_channels = 64 * 2 *2, output_channels = 64 * 2 * 2 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.final_fc = nn.Linear(16, 1)
        #self.final_pool = nn.AdaptiveAvgPool2d(1)   
        self.linear_mean = nn.Linear(512+13, 128)
        self.linear_std = nn.Linear(512+13, 128)
        # classification branch
        self.classification_branch  = Classification_Branch(input_dim = 512)
        
    def forward(self, x , c):
        x = self.spec_bn(x.unsqueeze(1))                        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print(x.shape)
        x = self.final_fc(x.flatten(-2))
        #print(x.shape)
        #x = self.final_pool(x)
        #classification
        
        class_pred = self.classification_branch(x)
        #print("class_info" ,class_info.shape) # output shape : torch.Size([16, 13])
        #c = (F.one_hot(c, num_classes=13)).unsqueeze(-1).unsqueeze(-1)
        #print("one_hot", c.shape) # shape: torch.Size([16, 13, 1, 1])
        #x = torch.cat([x, c], 1)#add class information
        #print(x.shape , class_pred.unsqueeze(-1).shape)
        x = torch.cat([x, class_pred.unsqueeze(-1)], 1)#add class information
        #print(x.squeeze(-1).squeeze(-1).shape)
        ## disentagle
        return self.linear_mean(x.squeeze(-1).squeeze(-1)), self.linear_std(x.squeeze(-1).squeeze(-1)), class_pred


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

class decoder_conv_block(nn.Module):
        def __init__(self, input_channels, output_channels):
            super(decoder_conv_block, self).__init__()
            self.conv_block1_1 = Con2d_block(input_channels = input_channels, output_channels = output_channels, kernel_size=3, stride=1, padding=1, pooling=1)
            self.conv_block1_2 = Con2d_block(input_channels = output_channels, output_channels = output_channels, kernel_size=3, stride=1, padding=1, pooling=1)
            self.conv_block1_3 = Con2d_block(input_channels = output_channels, output_channels = output_channels, kernel_size=3, stride=1, padding=1, pooling=1)
            self.conv_block1_4 = Con2d_block(input_channels = output_channels, output_channels = output_channels, kernel_size=3, stride=1, padding=1, pooling=1)

        def forward(self, x):
            x = self.conv_block1_1(x)
            x_residual = self.conv_block1_2(x)
            x_residual = self.conv_block1_3(x_residual)
            x_residual = self.conv_block1_4(x_residual)        
            return x + x_residual




class ImageDecoder(nn.Module):
    def __init__(self):
        '''
        input: latent z torch.Size([1, 128])
        output:  RGB image / torch.Size([3, 256, 256])
        '''
        super(ImageDecoder, self).__init__()
        self.de_block1 = decoder_block(128+13 , 128)
        self.conv_block1_1 = decoder_conv_block(input_channels = 128, output_channels = 128)
        self.conv_block1_2 = decoder_conv_block(input_channels = 128, output_channels = 128)
        self.de_block2 = decoder_block(128 , 128)
        self.conv_block2_1 = decoder_conv_block(input_channels = 128, output_channels = 64)
        self.conv_block2_2 = decoder_conv_block(input_channels = 64, output_channels = 64)
        # do upscaling
        self.Upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   
        self.de_block3 = decoder_block(64 , 64)
        self.conv_block3_1 = decoder_conv_block(input_channels = 64, output_channels = 32)
        self.conv_block3_2 = decoder_conv_block(input_channels = 32, output_channels = 32)
        
        self.de_block4 = decoder_block(32 , 32)
        self.conv_block4_1 = decoder_conv_block(input_channels = 32, output_channels = 16)
        self.conv_block4_2 = decoder_conv_block(input_channels = 16, output_channels = 16)
        # do upscaling
        self.Upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.de_block5 = decoder_block(16 , 16)
        self.conv_block5_1 = decoder_conv_block(input_channels = 16, output_channels = 16)
        #self.de_block6 = decoder_block(16 , 16)
        
        self.conv_block5_2 = decoder_conv_block(input_channels = 16, output_channels = 16)
        self.last_conv = nn.Conv2d(16, 3, kernel_size= 3, padding= 1)

            
    def forward(self, x, class_pred):
        ## disentagle
        #c = F.one_hot(c, num_classes=13)
        x = torch.cat([x, class_pred], 1)#add class information
        x = self.de_block1(x.unsqueeze(-1).unsqueeze(-1)) # torch.Size([1, 128, 2, 2])
        x = self.conv_block1_1(x)
        x = self.conv_block1_2(x)

        x = self.de_block2(x) # torch.Size([1, 64, 4, 4])
        x = self.conv_block2_1(x)
        x = self.conv_block2_2(x)

        x = self.Upsample1(x) # torch.Size([1, 64, 8, 8])
       
        x = self.de_block3(x) # torch.Size([1, 32, 16, 16])
        x = self.conv_block3_1(x)
        x = self.conv_block3_2(x)

        x = self.de_block4(x) # torch.Size([1, 16, 32, 32])
        x = self.conv_block4_1(x)
        x = self.conv_block4_2(x)

        x = self.Upsample2(x) # torch.Size([1, 16, 64, 64])
        x = self.de_block5(x) # torch.Size([1, 8, 128, 128])
        #x = self.de_block6(x) # torch.Size([1, 3, 256, 256])
        x = self.conv_block5_1(x)
        x = self.conv_block5_2(x)
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
        mean, logvar, class_pred = self.AudioEncoder(x, c)
        latent = self.sampling(mean, logvar)
        out = self.ImageDecoder(latent, class_pred)
        return out, mean, logvar, class_pred # torch.Size([3, 256, 256])
    
    
    
    
    
    