import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class ImageDecoder(nn.Module):
  '''
  Decode images from vectors. Similar structure as DCGAN.
  '''
  def __init__(self, input_size, n_channels, ngf, n_layers, activation='sigmoid'):
    super(ImageDecoder, self).__init__()

    ngf = ngf * (2 ** (n_layers - 2))
    layers = [nn.ConvTranspose2d(input_size, ngf, 4, 1, 0, bias=False),
              # nn.BatchNorm2d(ngf),
              nn.ReLU(True)]

    for i in range(1, n_layers - 1):
      layers += [nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
                 # nn.BatchNorm2d(ngf // 2),
                 nn.ReLU(True)]
      ngf = ngf // 2

    # layers += [nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
    #            # nn.BatchNorm2d(ngf // 2),
    #            nn.ReLU(True)]
    # layers += [nn.ConvTranspose2d(ngf, ngf, 1, 1, 0, bias=False),
    #            # nn.BatchNorm2d(ngf // 2),
    #            nn.ReLU(True)]
    layers += [nn.ConvTranspose2d(ngf, n_channels, 2, 2, 0, bias=False)]
    # layers += [nn.ConvTranspose2d(ngf, n_channels, 1, 1, 0, bias=False)]

    if activation == 'tanh':
      layers += [nn.Tanh()]
    elif activation == 'sigmoid':
      layers += [nn.Sigmoid()]
    else:
      raise NotImplementedError

    # self.main = nn.Sequential(*layers)

    self.main = nn.Sequential(
      nn.Linear(input_size, 256),  # B, 256
      # nn.ReLU(True),
      # nn.Linear(128, 256),  # B, 256
      View((-1, 256, 1, 1)),  # B, 256,  1,  1
      nn.ReLU(True),
      nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
      nn.ReLU(True),
      nn.ConvTranspose2d(32, n_channels, 4, 2, 1),  # B, nc, 64, 64
      # nn.Sigmoid()
    )

  def forward(self, x):
    # if len(x.size()) == 2:
    #   x = x.view(*x.size(), 1, 1)
    x = self.main(x)
    return x
