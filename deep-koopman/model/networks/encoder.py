import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class ImageEncoder(nn.Module):
    '''
    Encodes images. Similar structure as DCGAN.
    '''

    def __init__(self, in_channels, feat_dim, ngf, n_layers):
        super(ImageEncoder, self).__init__()

        layers = [nn.Conv2d(in_channels, ngf, 4, 2, 1, bias=False),
                  nn.LeakyReLU(0.2, inplace=True)]  # nn.LeakyReLU(0.2, inplace=True)]

        for i in range(1, n_layers - 1):
            layers += [nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
                       # nn.BatchNorm2d(ngf * 2),
                       nn.LeakyReLU(0.2, inplace=True)]
                       # nn.ReLU(True)]
            ngf *= 2

        # layers = [nn.Conv2d(in_channels, ngf, 4, 2, 1, bias=False),
        #           nn.ReLU(inplace=True)]
        #
        # for i in range(1, n_layers - 1):
        #     layers += [nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
        #                nn.BatchNorm2d(ngf * 2),
        #                nn.ReLU(inplace=True)]
        #     ngf *= 2

        layers += [nn.Conv2d(ngf, feat_dim, 4, 1, 0, bias=False),
                   # nn.BatchNorm2d(feat_dim),
                   nn.LeakyReLU(0.2, inplace=True)]
                   # nn.ReLU(True)]  # nn.LeakyReLU(0.2, inplace=True)]

        # layers += [nn.Conv2d(ngf, feat_dim, 4, 1, 0, bias=False)]

        # self.main = nn.Sequential(*layers)

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, feat_dim, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            # View((-1, 256*1*1)),                 # B, 256
            # nn.Linear(256, feat_dim),             # B, z_dim*2
        )
    def forward(self, x):

        x = self.main(x)
        x = x.squeeze(3).squeeze(2)

        return x
