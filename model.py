import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        #torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=2048, hop_length=512, f_min=0.0, f_max=8000.0, n_mels=128),
        #torchaudio.transforms.AmplitudeToDB()
        '''
        input: audio wave / after mel -> torch.Size([128, 44])
        output: latent vector
        
        '''
    def forward(self, x):
        
        #after mel -> torch.Size([128, 44])

        out = x
        return out

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        
        '''
        input: RGB image / torch.Size([3, 256, 256])
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
        output:  RGB image / torch.Size([3, 256, 256])
        
        '''

    def forward(self, x):

        out = x
        return out


class Audio2ImageVAE(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()

        '''
        input: audio wave
        output:  RGB image / torch.Size([3, 256, 256])
        
        '''

    def forward(self, x):

        latent = AudioEncoder(x)
        #addtional sampling
        out = ImageDecoder(latent)   
        return out # torch.Size([3, 256, 256])
