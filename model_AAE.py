import torchaudio

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classification_Branch(nn.Module):
    def __init__(self, input_dim = 128, output_class = 13):
        super(Classification_Branch, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.linear_last = nn.Linear(128, output_class)
        #self.LogSoftMax = nn.LogSoftmax()
      
    def forward(self, x):
        #print(": ", x.shape)
        x = self.relu1(self.linear1(x.squeeze(-1).squeeze(-1)))
        x = self.relu2(self.linear2(x))
        #x = self.LogSoftMax(self.linear_last(x))
        x = self.linear_last(x)
        return x


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
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Audio_block(input_channels = 1, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=4)
        self.layer2 = Audio_block(input_channels = 64, output_channels = 64 * 2, kernel_size=3, stride=1, padding=1, pooling=3)
        self.layer3 = Audio_block(input_channels = 64 * 2, output_channels = 64 * 2, kernel_size=3, stride=1, padding=1, pooling=3)

        self.final_pool = nn.AdaptiveAvgPool2d(1)   
        self.linear = nn.Linear(128+13, 128)
        # classification branch
        self.classification_branch  = Classification_Branch()
        
    def forward(self, x , c):
        x = self.spec_bn(x.unsqueeze(1))                        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_pool(x)        
        class_pred = self.classification_branch(x)
        x = torch.cat([x, class_pred.unsqueeze(-1).unsqueeze(-1)], 1)#add class information
        return self.linear(x.squeeze(-1).squeeze(-1)), class_pred


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

            
    def forward(self, x, class_pred):
        ## disentagle
        #c = F.one_hot(c, num_classes=13)
        x = torch.cat([x, class_pred], 1)#add class information
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

#============================NEW EMBEDDING AVE==================================================

def getNonLinear(nlu):

    if nlu == "relu":
       return nn.ReLU(inplace=True)
    elif nlu == "glu": #gated linear unit
       return F.glu
    else:
       return None
   
def getNorm(norm):

    if norm =="bn": #default batchnorm
       return nn.BatchNorm2d
    else:
       return None
   
def Conv3x3(in_ch, out_ch, stride):
    """ 3x3 convolution with padding """
    conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, 
                     padding=1)
    
    return conv

def Conv1x1(in_ch, out_ch, stride=1):
    """ 1x1 convolution """
    conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride, bias=False)
    return conv

class BasicBlock(nn.Module):
  """ Basic conv block"""
  
  def __init__(self, in_ch, out_ch, stride, ps, norm):
    
      super(BasicBlock, self).__init__()

      norm_fn = getNorm(norm)  

      self.conv1 = Conv3x3(in_ch, out_ch, stride)
      self.bn1 = norm_fn(out_ch)
        
      self.conv2 = Conv3x3(out_ch, out_ch, stride=1)
      self.bn2 = norm_fn(out_ch)

      self.relu = getNonLinear("relu")

  def forward(self, x):
    
      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)
      out = self.relu(out)

      return out

class AudNet(nn.Module):
  """ Audio subnet """

  def __init__(self, norm, init_wts=True):

        super(AudNet, self).__init__()

        out_ch = [64, 128, 256, 512]
        #for spectrogram features as mentioned in Objects that Sound paper
        self.c1 = BasicBlock(in_ch=1, out_ch=out_ch[0], stride=2, ps=2) 
        self.c2 = BasicBlock(in_ch=out_ch[0], out_ch=out_ch[1], stride=1, ps=2, norm=norm)
        self.c3 = BasicBlock(in_ch=out_ch[1], out_ch=out_ch[2], stride=1, ps=2, norm=norm)
        self.c4 = BasicBlock(in_ch=out_ch[2], out_ch=out_ch[3], stride=1, ps=2, norm=norm)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_final = nn.MaxPool2d(kernel_size=(16, 12), stride=(16, 12))


        fc_dim = 128
        self.norm = F.normalize
        self.embed = nn.Sequential(
            nn.Linear(512, fc_dim),
            getNonLinear("relu"),
            nn.Linear(fc_dim+13, fc_dim)
            )

        self.classification_branch  = Classification_Branch()
      
  def forward(self, x):
     
        #conv features
        out = self.c1(x)
        #print("c1 features", out.shape)
        out = self.pool1(out)
        #print("p1 features", out.shape)
        out = self.c2(out)
        #print("c2 features", out.shape)
        out = self.pool(out)
        #print("p2 features", out.shape)
        out = self.c3(out)
        #print("c3 features", out.shape)
        out = self.pool(out)
        #print("p3 features", out.shape)
        out = self.c4(out)
        #print("c4 features", out.shape)
        out = self.pool_final(out)
        #print("aud features:", out.shape)

        out = out.view(out.shape[0], -1)
        #print("flatten:", out.shape)

        #conv embeddings
        out = self.embed(out)
        #print("embed:", out.shape)

        #normalize embeddings to length=1
        out = self.norm(out, p=2, dim=1)
        #print("norm:", out.shape)

        class_pred = self.classification_branch(out)
        out = torch.cat([out, class_pred.unsqueeze(-1).unsqueeze(-1)], 1)#add class information
        
        return self.linear(x.squeeze(-1).squeeze(-1)), class_pred

#+==================================================================================


class Audio2ImageCVAE(nn.Module):
    def __init__(self):
        super(Audio2ImageCVAE, self).__init__()
        '''
        input: audio wave
        output:  RGB image / torch.Size([3, 256, 256])
        '''
        #self.AudioEncoder = AudioEncoder()
        self.AudioEncoder = AudNet(norm="bn")
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
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 64)        
        self.lin3 = nn.Linear(64, 1)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)        
        self.sigmo = nn.Sigmoid()
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.lin3(x)
        return self.sigmo(x)  