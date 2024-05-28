import torch
import torch.nn as nn

class decoder(nn.Module):
    def __init__(self, encoder_hidden_dims=24, decoder_hidden_dims=512):
        super(decoder, self).__init__()
        self.deconder1 = torch.nn.Conv2d(in_channels=encoder_hidden_dims, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.deconder2 = torch.nn.Conv2d(in_channels=128, out_channels=256,
                                         kernel_size=(1, 1), stride=(1, 1))
        self.deconder3 = torch.nn.Conv2d(in_channels=256, out_channels=decoder_hidden_dims,
                                         kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x):
        x = self.deconder1(x)
        x = self.deconder2(x)
        x = self.deconder3(x)
        return x