import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImageDecoder(nn.Module):
  def __init__(self, input_size, out_ch, dyn_dim):
    super(ImageDecoder, self).__init__()

    self.dyn_dim = dyn_dim
    init_ch = 256
    self.init_dec_cnn = nn.Sequential(
      nn.Conv2d(input_size, init_ch, 1),
      nn.CELU(),
      nn.BatchNorm2d(init_ch),
    )

    # TO X CNN Maps appearance to object. 1 x 1
    self.to_x = nn.Sequential(
      nn.Conv2d(init_ch, 128 * 2 * 2, 1), # 2, 2
      nn.PixelShuffle(2),
      nn.CELU(),
      nn.BatchNorm2d(128),
      nn.Conv2d(128, 128, 3, 1, 1),
      nn.CELU(),
      nn.BatchNorm2d(128),

      nn.Conv2d(128, 128 * 2 * 2, 1), # 4, 4
      nn.PixelShuffle(2),
      nn.CELU(),
      nn.BatchNorm2d(128),
      nn.Conv2d(128, 128, 3, 1, 1),
      nn.CELU(),
      nn.BatchNorm2d(128),
      # --
      nn.Conv2d(128, 64 * 2 * 2, 1), # 8, 8
      nn.PixelShuffle(2),
      nn.CELU(),
      nn.BatchNorm2d(64),
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.CELU(),
      nn.BatchNorm2d(64),
      # --
      nn.Conv2d(64, 32 * 2 * 2, 1), # 16, 16
      nn.PixelShuffle(2),
      nn.CELU(),
      nn.BatchNorm2d(32),
      nn.Conv2d(32, 32, 3, 1, 1),
      nn.CELU(),
      nn.BatchNorm2d(32),
      # --
      nn.Conv2d(32, 16 * 2 * 2, 1), # 32, 32
      nn.PixelShuffle(2),
      nn.CELU(),
      nn.BatchNorm2d(16),
      nn.Conv2d(16, 16, 3, 1, 1),
      nn.CELU(),
      nn.BatchNorm2d(16),
      # --
      nn.Conv2d(16, out_ch, 3, 1, 1) # output channels. If your input is RGB, out_ch=3
    )

  def forward(self, input, block = 'all'):
    bs, dim = input.shape
    input = input.reshape(*input.size(), 1, 1)
    x = self.init_dec_cnn(input)

    if block == 'to_x':
      x = torch.sigmoid(self.to_x(x)) # output will be [0, 1]

    return x

class ImageBroadcastDecoder(nn.Module): # lots of memory
  def __init__(self, input_size, out_ch, resolution=(32, 32)):
    super(ImageBroadcastDecoder, self).__init__()

    # at the beginning bs, T, chan, 8 x 8 --> 128 x 128
    self.to_x = nn.Sequential(
      nn.Conv2d(input_size + 4, 128, 1), # 8, 8
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.Conv2d(128, 128, 3, 1, 1),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      # --
      nn.Conv2d(128, 64 * 2 * 2, 1),
      nn.PixelShuffle(2),
      nn.ReLU(),
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      # --
      nn.BatchNorm2d(64),
      nn.Conv2d(64, 64 * 2 * 2, 1),
      nn.PixelShuffle(2),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.Conv2d(64, 32, 3, 1, 1),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      # --
      nn.Conv2d(32, 32 * 2 * 2, 1), # 128, 128
      nn.PixelShuffle(2),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.Conv2d(32, 16, 3, 1, 1),
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.Conv2d(16, out_ch, 1)
    )

    self.hard_pe = True
    self.decoder_initial_size = resolution

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

  def forward(self, input, block = 'all'):
    bs, dim = input.shape
    # input is appearance vector. You will predict the location of the object with only the appearance vector.
    # Read spatial broadcast paper
    if block == 'to_x':
      x = spatial_broadcast(input, self.decoder_initial_size)

      if self.hard_pe:
        # Hard PE
        x = torch.cat([x, self.grid_dec.repeat(bs, 1, 1)], dim=-1)
      else:
        # Soft PE
        x = self.decoder_pos(x)

      x = x.permute(0,2,1).reshape(bs, -1, *self.decoder_initial_size)
      x = torch.sigmoid(self.to_x(x))

    return x

    # Undo combination of slot and batch dimension; split alpha masks.
    # recons, masks = unstack_and_split(x, batch_size=b)
    # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
    # `masks` has shape: [batch_size, num_slots, width, height, 1].

    # Normalize alpha masks over slots.
    # masks = F.softmax(masks, dim=1)

    # recon_combined = torch.reduce_sum(recons * masks, axis=1)  # Recombine image.
    # `recon_combined` has shape: [batch_size, width, height, num_channels].

def spatial_broadcast(slots, resolution):
  """Broadcast slot features to a 2D grid and collapse slot dimension."""
  # `slots` has shape: [batch_size, num_slots, slot_size].
  slots = torch.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
  grid = slots.repeat(1, resolution[0], resolution[1], 1).reshape(-1, resolution[0]*resolution[1], slots.shape[-1])
  # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
  return grid

