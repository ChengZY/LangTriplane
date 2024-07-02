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

class decoder_render(nn.Module):
    def __init__(self, encoder_hidden_dims=27, decoder_hidden_dims=512): # 24 channel 3 image 1 edge
        super(decoder_render, self).__init__()
        self.deconder1 = torch.nn.Conv2d(in_channels=encoder_hidden_dims, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.deconder2 = torch.nn.Conv2d(in_channels=128, out_channels=256,
                                         kernel_size=(1, 1), stride=(1, 1))
        self.deconder3 = torch.nn.Conv2d(in_channels=256, out_channels=decoder_hidden_dims,
                                         kernel_size=(1, 1), stride=(1, 1))

        # nn.init.constant_(self.deconder1.weight, 0.00001)
        # nn.init.constant_(self.deconder2.weight, 0.00001)
        # nn.init.constant_(self.deconder3.weight, 0.00001)
        # if self.deconder1.bias is not None:
        #     nn.init.constant_(self.deconder1.bias, 0.00001)
        # if self.deconder2.bias is not None:
        #     nn.init.constant_(self.deconder2.bias, 0.00001)
        # if self.deconder3.bias is not None:
        #     nn.init.constant_(self.deconder3.bias, 0.00001)


    def forward(self, language_feature_image, render_image):
        language_feature_image = torch.cat([language_feature_image, render_image])
        language_feature_image = self.deconder1(language_feature_image)
        language_feature_image = self.deconder2(language_feature_image)
        language_feature_image = self.deconder3(language_feature_image)

        # render_image = self.render_encoder1(render_image)
        # render_image = self.render_encoder2(render_image)
        # render_image = self.render_encoder3(render_image)

        # language_feature_image = self.render_encoder4(torch.cat([language_feature_image, render_image]))

        return language_feature_image

def init_weights_to_one(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.00001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.00001)