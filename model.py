import torch
import torch.nn as nn
import torch.nn.functional as F

# Auxiliary Classifier
class Classification_Branch(nn.Module):
    def __init__(self, input_dim = 128, output_class = 13):
        super(Classification_Branch, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2 )
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_dim // 2, input_dim // 4 )
        self.relu2 = nn.ReLU()
        self.linear_last = nn.Linear(input_dim // 4, output_class)
      
    def forward(self, x):
        x = self.relu1(self.linear1(x.squeeze(-1).squeeze(-1)))
        x = self.relu2(self.linear2(x))
        x = self.linear_last(x)
        return x

# Convolution Module
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

# Discriminator
class ImageDiscrimitor(nn.Module):
    def __init__(self):
        super(ImageDiscrimitor, self).__init__()
        self.layer1 = Con2d_block(input_channels = 3, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer2 = Con2d_block(input_channels = 64, output_channels = 64 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer3 = Con2d_block(input_channels = 64 * 2, output_channels = 64 * 2 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer4 = Con2d_block(input_channels = 64 * 2 * 2, output_channels = 64 * 2 * 2 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer5 = Con2d_block(input_channels = 64 * 2 * 2 * 2, output_channels = 64 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.final_fc1 = nn.Linear(128 * 4* 4, 64)
        self.final_fc2 = nn.Linear(64, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.final_fc1(x.flatten(-3))
        x = self.final_fc2(x)
        return self.sigmoid_layer(x)
            
class AudioDiscrimitor(nn.Module):
    def __init__(self):
        super(AudioDiscrimitor, self).__init__()
        self.layer1 = Con2d_block(input_channels = 1, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer2 = Con2d_block(input_channels = 64, output_channels = 64 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer3 = Con2d_block(input_channels = 64 * 2, output_channels = 64 * 2 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer4 = Con2d_block(input_channels = 64 * 2 *2, output_channels = 64 * 2 * 2 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.final_fc1 = nn.Linear(16, 1)       
        self.final_fc2 = nn.Linear(512, 256)
        self.final_fc3 = nn.Linear(256, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):      
        x = self.layer1(x.unsqueeze(1))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_fc1(x.flatten(-2))
        x = self.final_fc2(x.squeeze(-1))
        x = self.final_fc3(x)
        return self.sigmoid_layer(x) 
        
# Encoder
class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.spec_bn = nn.BatchNorm2d(1)
        self.layer1 = Con2d_block(input_channels = 1, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer2 = Con2d_block(input_channels = 64, output_channels = 64 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer3 = Con2d_block(input_channels = 64 * 2, output_channels = 64 * 2 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer4 = Con2d_block(input_channels = 64 * 2 *2, output_channels = 64 * 2 * 2 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.final_fc = nn.Linear(16, 1)
        self.linear_mean = nn.Linear(512+13, 128)
        self.linear_std = nn.Linear(512+13, 128)
        self.classification_branch  = Classification_Branch(input_dim = 512)
        
    def forward(self, x , c):
        x = self.spec_bn(x.unsqueeze(1))                        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_fc(x.flatten(-2))

        class_pred = self.classification_branch(x)
        x = torch.cat([x, class_pred.unsqueeze(-1)], 1) #add class information

        return self.linear_mean(x.squeeze(-1).squeeze(-1)), self.linear_std(x.squeeze(-1).squeeze(-1)), class_pred 
        
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.layer1 = Con2d_block(input_channels = 3, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer2 = Con2d_block(input_channels = 64, output_channels = 64 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer3 = Con2d_block(input_channels = 64 * 2, output_channels = 64 * 2 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer4 = Con2d_block(input_channels = 64 * 2 * 2, output_channels = 64 * 2 * 2 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.layer5 = Con2d_block(input_channels = 64 * 2 * 2 * 2, output_channels = 64 * 2 * 2 * 2, kernel_size=3, stride=1, padding=1, pooling=2)
        self.final_fc = nn.Linear(16, 1)
        self.linear_mean = nn.Linear(512+13, 128)
        self.linear_std = nn.Linear(512+13, 128)
        self.classification_branch  = Classification_Branch(input_dim = 512)
    def forward(self, x , c):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.final_fc(x.flatten(-2))
        class_pred = self.classification_branch(x)
        x = torch.cat([x, class_pred.unsqueeze(-1)], 1)
        return self.linear_mean(x.squeeze(-1).squeeze(-1)), self.linear_std(x.squeeze(-1).squeeze(-1)), class_pred

# Decoder
class AudioDecoder(nn.Module):
    def __init__(self):
        super(AudioDecoder, self).__init__()
        self.de_block1 = decoder_block(128+13 , 128)
        self.conv_block1_1 = decoder_conv_block(input_channels = 128, output_channels = 128)
        self.conv_block1_2 = decoder_conv_block(input_channels = 128, output_channels = 128)
        
        self.de_block2 = decoder_block(128 , 128)
        self.conv_block2_1 = decoder_conv_block(input_channels = 128, output_channels = 64)
        self.conv_block2_2 = decoder_conv_block(input_channels = 64, output_channels = 64)
        
        self.Upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   
        self.de_block3 = decoder_block(64 , 64)
        self.conv_block3_1 = decoder_conv_block(input_channels = 64, output_channels = 32)
        self.conv_block3_2 = decoder_conv_block(input_channels = 32, output_channels = 32)
        
        self.de_block4 = decoder_block(32 , 32)
        self.conv_block4_1 = decoder_conv_block(input_channels = 32, output_channels = 16)
        self.conv_block4_2 = decoder_conv_block(input_channels = 16, output_channels = 16)
        
        self.Upsample2 = nn.Upsample(size=(64,44), mode='bilinear', align_corners=True)  
        self.conv_block5_1 = decoder_conv_block(input_channels = 16, output_channels = 16)
        self.Upsample3 = nn.Upsample(size=(96,44), mode='bilinear', align_corners=True)  
        self.conv_block5_2 = decoder_conv_block(input_channels = 16, output_channels = 16)

        self.Upsample4 = nn.Upsample(size=(128,44), mode='bilinear', align_corners=True)  
        self.last_conv1 = nn.Conv2d(16, 64, kernel_size= 3, padding= 1)
        self.last_conv2 = nn.Conv2d(64, 32, kernel_size= 3, padding= 1)
        self.last_conv3 = nn.Conv2d(32, 1, kernel_size= 3, padding= 1)

            
    def forward(self, x, class_pred):
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
        x = self.conv_block5_1(x)
        x = self.Upsample3(x) 
        x = self.conv_block5_2(x)

        x = self.Upsample4(x)
        x = self.last_conv1(x)
        x = self.last_conv2(x)
        x = self.last_conv3(x)
        
        return x.squeeze(1)

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
        
        self.conv_block5_2 = decoder_conv_block(input_channels = 16, output_channels = 16)
        self.last_conv = nn.Conv2d(16, 3, kernel_size= 3, padding= 1)

            
    def forward(self, x, class_pred):
        x = torch.cat([x, class_pred], 1)
        
        x = self.de_block1(x.unsqueeze(-1).unsqueeze(-1)) 
        x = self.conv_block1_1(x)
        x = self.conv_block1_2(x)

        x = self.de_block2(x)
        x = self.conv_block2_1(x)
        x = self.conv_block2_2(x)

        x = self.Upsample1(x) 
       
        x = self.de_block3(x)
        x = self.conv_block3_1(x)
        x = self.conv_block3_2(x)

        x = self.de_block4(x)
        x = self.conv_block4_1(x)
        x = self.conv_block4_2(x)
        
        x = self.Upsample2(x)
        
        x = self.de_block5(x) 
        x = self.conv_block5_1(x)
        x = self.conv_block5_2(x)
        
        x = self.last_conv(x)
        return x

# Conditional VAE
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
        
class Image2AudioCVAE(nn.Module):
    def __init__(self):
        super(Image2AudioCVAE, self).__init__()
        '''
        input: audio wave
        output:  RGB image / torch.Size([3, 256, 256])
        '''
        self.ImageEncoder = ImageEncoder()
        self.AudioDecoder = AudioDecoder()

    def sampling(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, x, c):
        mean, logvar, class_pred = self.ImageEncoder(x, c)
        latent = self.sampling(mean, logvar)
        out = self.AudioDecoder(latent, class_pred)
        return out, mean, logvar, class_pred # torch.Size([3, 256, 256])
    
    
class CrossModal(nn.Module):
    def __init__(self):
        super(CrossModal, self).__init__()
        # Audio2ImageCVAE
        self.AudioEncoder = AudioEncoder()
        self.ImageDecoder = ImageDecoder()
        # Image2AudioCVAE
        self.ImageEncoder = ImageEncoder()
        self.AudioDecoder = AudioDecoder() 
        
    def sampling(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, image, lms, c):
        # Audio -> Image
        mean_a, logvar_a, class_pred_a = self.AudioEncoder(lms, c)
        latent_a = self.sampling(mean_a, logvar_a)
        out_image = self.ImageDecoder(latent_a, class_pred_a)
        # Image -> Audio
        mean_v, logvar_v, class_pred_v = self.ImageEncoder(image, c)
        latent_v = self.sampling(mean_v, logvar_v)
        out_lms = self.AudioDecoder(latent_v, class_pred_v)
        
        return out_image, latent_a, mean_a, logvar_a, class_pred_a, out_lms, latent_v, mean_v, logvar_v, class_pred_v     