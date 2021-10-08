import torch.nn as nn
import torch

class BackgroundAE(nn.Module):
    def __init__(self, in_channels):
        super(BackgroundAE, self).__init__()
        self.encoder = BgEncoder(in_channels)
        self.decoder = BgDecoder(in_channels)
    def forward(self, x):
        return self.decoder(self.encoder(x))

class BgEncoder(nn.Module):

    def __init__(self, in_channels):
        super(BgEncoder, self).__init__()

        layers = [nn.Conv2d(in_channels, 16, 4, 2, 1), # 64, 64
                  nn.CELU(), # use ReLU
                  nn.GroupNorm(4, 16), # not use batchnorm
                  nn.Conv2d(16, 32, 4, 2, 1),
                  # nn.GroupNorm(4, 16), # not use batchnorm
                  # nn.Conv2d(16, 32, 4, 2, 1),
                  nn.CELU(),
                  nn.GroupNorm(4, 32),
                  nn.Conv2d(32, 64, 7, 1, 3),
                  ]

        self.cnn_bg = nn.Sequential(*layers)
        # self.lstm_bg = nn.LSTM(32, 32, batch_first=True, num_layers=1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = self.cnn_bg(x.reshape(-1, *x.shape[2:])) # TODO: Encode with lateral encoder ? #convolutional encoder
        (c, h, w) = x.shape[-3:]
        x = x.reshape(B, T, c, h*w).permute(0, 3, 1, 2).reshape(-1, T, c)

        # Option 1: maxpool
        x = torch.max(x, dim=1)[0]
        enc_bg = x.reshape(B, h, w, c).permute(0, 3, 1, 2)

        # Option 2: lstm
        # x, _ = self.lstm_bg(x) # TODO: Add fc
        # enc_bg = x[:, -1].reshape(B, h, w, c).permute(0, 3, 1, 2)

        return enc_bg

class BgDecoder(nn.Module):

    def __init__(self, out_channels):
        super(BgDecoder, self).__init__()

        layers = [nn.Conv2d(64, 32, 5, 1, 2), # 64, 64
                  nn.UpsamplingBilinear2d(scale_factor=2),
                  nn.CELU(),
                  nn.GroupNorm(4, 32),
                  nn.Conv2d(32, 16, 5, 1, 2),
                  nn.UpsamplingBilinear2d(scale_factor=2),
                  nn.CELU(),
                  nn.GroupNorm(4, 16),
                  nn.Conv2d(16, out_channels, 5, 1, 2)]

        self.cnn_bg = nn.Sequential(*layers)

    def forward(self, x):
        # bs, c, h, w = x.shape
        out = self.cnn_bg(x)[:, None].sigmoid()

        return out