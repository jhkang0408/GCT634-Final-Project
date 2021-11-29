import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        

        '''
        input: audio wave
        output: latent vector
        
        '''
    def forward(self, x):

        out = x
        return out

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        
        '''
        input: RGB image
        output: latent vector
        
        '''
    def forward(self, x):

        out = x
        return out


class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()
        '''
        input: latent vector
        output:  RGB image
        
        '''

    def forward(self, x):

        out = x
        return out

