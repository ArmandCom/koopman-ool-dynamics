import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSBImageDecoder(nn.Module):
  '''
  Decode images from vectors. Similar structure as DCGAN.
  '''
  def __init__(self, feat_dim, out_channels, ngf, n_layers, im_size, activation='sigmoid'): # (self, n_channels, output_size, ngf, n_layers)
    super(SimpleSBImageDecoder, self).__init__()

    # Coordinates for the broadcast decoder
    self.im_size = im_size
    x = torch.linspace(-1, 1, im_size[0])
    y = torch.linspace(-1, 1, im_size[1])
    x_grid, y_grid = torch.meshgrid(x, y)
    # Add as constant, with extra dims for N and C
    self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
    self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

    ngf = 64
    layers = [nn.Conv2d(feat_dim + 2, ngf, 3, 1, 1, bias=False), # feat_dim + 2 (coord_dim)
              nn.BatchNorm2d(ngf),
              nn.ReLU(True)] # nn.LeakyReLU(0.2, inplace=True)]

    for i in range(2):
        layers += [nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                   nn.BatchNorm2d(ngf),
                   nn.ReLU(True)]  # nn.LeakyReLU(0.2, inplace=True)]

    layers += [nn.Conv2d(ngf, out_channels, 3, 1, 1, bias=False)]

    if activation == 'tanh':
      layers += [nn.Tanh()]
    elif activation == 'sigmoid':
      layers += [nn.Sigmoid()]
    else:
      raise NotImplementedError

    self.conv_decoder = nn.Sequential(*layers)
    self.main = self.sb_decoder

  def sb_decoder(self, z):

      batch_size = z.size(0)

      # View z as 4D tensor to be tiled across new H and W dimensions
      # Shape: NxDx1x1
      z = z.view(z.shape + (1, 1))

      # Tile across to match image size
      # Shape: NxDx64x64
      z = z.expand(-1, -1, self.im_size[0], self.im_size[1])

      # Expand grids to batches and concatenate on the channel dimension
      # Shape: Nx(D+2)x64x64
      x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                     self.y_grid.expand(batch_size, -1, -1, -1), z), dim=1)

      x = self.conv_decoder(x)
      return x

  def forward(self, x):
    x = self.main(x)
    return x

class SBImageDecoder(nn.Module):
  '''
  Decode images from vectors. Similar structure as DCGAN.
  '''
  def __init__(self, feat_dim, out_channels, ngf, n_layers, im_size, activation='tanh'): # (self, n_channels, output_size, ngf, n_layers)
    super(SBImageDecoder, self).__init__()

    # Coordinates for the broadcast decoder
    self.im_size = im_size
    x = torch.linspace(-1, 1, im_size[0])
    y = torch.linspace(-1, 1, im_size[1])
    x_grid, y_grid = torch.meshgrid(x, y)
    # Add as constant, with extra dims for N and C
    self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
    self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

    ngf = ngf * (2 ** (n_layers - 2))

    # Note: Decoder with upsampling - without Spatial Broadcasting
    # layers = [nn.ConvTranspose2d(feat_dim, ngf, 4, 1, 0, bias=False),
    #           nn.BatchNorm2d(ngf),
    #           nn.ReLU(True)]
    # for i in range(1, n_layers - 1):
    #   layers += [nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
    #              nn.BatchNorm2d(ngf // 2),
    #              nn.ReLU(True)]
    #   ngf = ngf // 2
    # layers += [nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False)]

    layers = [nn.ConvTranspose2d(feat_dim + 2, ngf, 3, 1, 1, bias=False), # feat_dim + 2 (coord_dim)
              nn.BatchNorm2d(ngf),
              nn.ReLU(True)] # nn.LeakyReLU(0.2, inplace=True)]

    for i in range(1, 4): # Note: added to provide more depth
        layers += [nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=False),
                   nn.BatchNorm2d(ngf),
                   nn.ReLU(True)]  # nn.LeakyReLU(0.2, inplace=True)]

    for i in range(1, n_layers - 1):
        layers += [nn.ConvTranspose2d(ngf, ngf // 2, 3, 1, 1, bias=False),
                   nn.BatchNorm2d(ngf // 2),
                   nn.ReLU(True)]  # nn.LeakyReLU(0.2, inplace=True)]
        ngf = ngf // 2

    layers += [nn.ConvTranspose2d(ngf, out_channels, 3, 1, 1, bias=False)]
    if activation == 'tanh':
      layers += [nn.Tanh()]
    elif activation == 'sigmoid':
      layers += [nn.Sigmoid()]
    else:
      raise NotImplementedError

    self.conv_decoder = nn.Sequential(*layers)
    self.main = self.sb_decoder

  def sb_decoder(self, z):

      batch_size = z.size(0)

      # View z as 4D tensor to be tiled across new H and W dimensions
      # Shape: NxDx1x1
      z = z.view(z.shape + (1, 1))

      # Tile across to match image size
      # Shape: NxDx64x64
      z = z.expand(-1, -1, self.im_size[0], self.im_size[1])

      # Expand grids to batches and concatenate on the channel dimension
      # Shape: Nx(D+2)x64x64
      x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                     self.y_grid.expand(batch_size, -1, -1, -1), z), dim=1)

      x = self.conv_decoder(x)
      return x

  def forward(self, x):
    x = self.main(x)
    return x
