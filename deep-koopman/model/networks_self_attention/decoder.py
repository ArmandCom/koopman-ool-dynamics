import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .self_attention import Self_Attn

# class ImageDecoder(nn.Module):
#   """Generator."""
#
#   def __init__(self, input_size, n_channels, ngf, n_layers, activation='sigmoid'):
#     super(ImageDecoder, self).__init__()
#     self.imsize = 64
#     z_dim=input_size
#     conv_dim = 16
#
#     layer1 = []
#     layer2 = []
#     layer3 = []
#     last = []
#
#     repeat_num = int(np.log2(self.imsize)) - 3
#     mult = 2 ** repeat_num  # 8
#     layer1.append((nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
#     layer1.append(nn.BatchNorm2d(conv_dim * mult))
#     layer1.append(nn.ReLU())
#
#     curr_dim = conv_dim * mult
#
#     layer2.append((nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
#     layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
#     layer2.append(nn.ReLU())
#
#     curr_dim = int(curr_dim / 2)
#
#     self.attn1 = Self_Attn(curr_dim, 'relu')
#
#     layer3.append((nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
#     layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
#     layer3.append(nn.ReLU())
#
#     self.attn2 = Self_Attn(int(curr_dim / 2), 'relu')
#
#     if self.imsize == 64:
#       layer4 = []
#       curr_dim = int(curr_dim / 2)
#       layer4.append((nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
#       layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
#       layer4.append(nn.ReLU())
#       self.l4 = nn.Sequential(*layer4)
#       curr_dim = int(curr_dim / 2)
#
#     self.l1 = nn.Sequential(*layer1)
#     self.l2 = nn.Sequential(*layer2)
#     self.l3 = nn.Sequential(*layer3)
#
#     last.append(nn.ConvTranspose2d(curr_dim, n_channels, 4, 2, 1))
#     # last.append(nn.Sigmoid())
#     self.last = nn.Sequential(*last)
#
#   def forward(self, z):
#     z = z.view(z.size(0), z.size(1), 1, 1)
#     out = self.l1(z)
#     out = self.l2(out)
#     out, p2 = self.attn1(out)
#     out = self.l3(out)
#     out, p1 = self.attn2(out)
#     out = self.l4(out)
#     out = self.last(out)
#     return out
#     # return out, p1, p2

class ImageDecoder(nn.Module):
  def __init__(self, input_size, n_channels, ngf, n_layers, activation='sigmoid'):
    super(ImageDecoder, self).__init__()
    # TODO: added self attention layers (only) - not tested
    ch = 32
    self.main = nn.Sequential(
      nn.ConvTranspose2d(input_size, ch, 5, 1, 2),
      nn.ReLU(True),
      Self_Attn(ch, 'relu'),
      nn.ConvTranspose2d(ch, ch, 4, 2, 1),
      nn.ReLU(True),
      Self_Attn(ch, 'relu'),
      nn.ConvTranspose2d(ch, ch, 4, 2, 1),
      nn.ReLU(True),
      nn.ConvTranspose2d(ch, ch, 4, 2, 1),
      nn.ReLU(True),
      nn.ConvTranspose2d(ch, ch, 5, 1, 2),
      nn.ReLU(True),
      nn.ConvTranspose2d(ch, n_channels, 3, 1, 1),  # B, nc, 64, 64
      # nn.Sigmoid()
    )

    self.decoder_initial_size = (8, 8)

    # self.decoder_pos = SoftPositionEmbed(32, self.decoder_initial_size)

    self.linear_pos = nn.Linear(4, input_size)
    self.register_buffer('grid_dec', self._build_grid(self.decoder_initial_size))
    self.decoder_pos = self._soft_position_embed

  def _soft_position_embed(self, input):

    return input + self.linear_pos(self.grid_dec)

  def _build_grid(self, resolution):
      ranges = [np.linspace(0., 1., num=res) for res in resolution]
      grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
      grid = np.stack(grid, axis=-1)
      grid = np.reshape(grid, [resolution[0], resolution[1], -1])
      grid = np.expand_dims(grid, axis=0)
      grid = grid.astype(np.float32)
      return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).reshape(grid.shape[0], -1, 4)

  def forward(self, x):
    b, dim = x.shape
    # if len(x.size()) == 2:
    #   x = x.view(*x.size(), 1, 1)
    x = spatial_broadcast(x, self.decoder_initial_size)
    x = self.decoder_pos(x)
    x = x.permute(0,2,1).reshape(b, dim, *self.decoder_initial_size)
    x = self.main(x)

    # Undo combination of slot and batch dimension; split alpha masks.
    # recons, masks = unstack_and_split(x, batch_size=b)
    # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
    # `masks` has shape: [batch_size, num_slots, width, height, 1].

    # Normalize alpha masks over slots.
    # masks = F.softmax(masks, dim=1)

    # recon_combined = torch.reduce_sum(recons * masks, axis=1)  # Recombine image.
    # `recon_combined` has shape: [batch_size, width, height, num_channels].
    # print(x.shape)

    return x

def spatial_broadcast(slots, resolution):
  """Broadcast slot features to a 2D grid and collapse slot dimension."""
  # `slots` has shape: [batch_size, num_slots, slot_size].
  slots = torch.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
  grid = slots.repeat(1, resolution[0], resolution[1], 1).reshape(-1, resolution[0]*resolution[1], slots.shape[-1])
  # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
  return grid

class SoftPositionEmbed(nn.Module):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, hidden_size, resolution):
    """Builds the soft position embedding layer.
    Args:
      hidden_size: Size of input feature dimension.
      resolution: Tuple of integers specifying width and height of grid.
    """
    super().__init__()
    self.dense = nn.Linear(hidden_size, hidden_size)
    self._build_grid(resolution)

  def forward(self, inputs):
    return inputs + self.dense(self.grid)

  def _build_grid(self, resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    self.grid = torch.tensor(np.concatenate([grid, 1.0 - grid], axis=-1)).reshape(grid.shape[0], -1, grid.shape[1])


def spatial_flatten(x):
  return torch.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])

class View(nn.Module):
  def __init__(self, size):
    super(View, self).__init__()
    self.size = size

  def forward(self, tensor):
    return tensor.view(self.size)