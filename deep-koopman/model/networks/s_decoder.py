from __future__ import print_function,division

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data

class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.Tanh):
        super(ResidLinear, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)


class deconvSpatialDecoder(nn.Module):
    def __init__(self, feat_dim, out_channels, hidden_dim, ngf, num_layers=1, activation=nn.Tanh
                , softplus=True, resid=False, expand_coords=False, bilinear=False):
        super(deconvSpatialDecoder, self).__init__()

        self.image_size = [64, 64]
        self.softplus = softplus
        self.expand_coords = expand_coords

        in_dim = 2
        if expand_coords:
            in_dim = 5 # include squares of coordinates as inputs

        self.coord_linear = nn.Linear(in_dim, hidden_dim)
        self.feat_dim = feat_dim
        if feat_dim > 0:
            self.latent_linear = nn.Linear(feat_dim, hidden_dim, bias=False)

        if feat_dim > 0 and bilinear: # include bilinear layer on latent and coordinates
            self.bilinear = nn.Bilinear(in_dim, feat_dim, hidden_dim, bias=False)

        #---------------------------------------------------------------------
        ngf = ngf * 8
        layers = [nn.Conv2d(hidden_dim, ngf, 3, 1, 1, bias=False),  # feat_dim + 2 (coord_dim)
                  nn.BatchNorm2d(ngf),
                  nn.ReLU(True)]  # nn.LeakyReLU(0.2, inplace=True)]

        for i in range(num_layers - 2):
            layers += [nn.Conv2d(ngf, ngf // 2, 3, 1, 1, bias=False),
                       nn.BatchNorm2d(ngf // 2),
                       nn.ReLU(True)]  # nn.LeakyReLU(0.2, inplace=True)]
            ngf = ngf // 2
        layers += [nn.Conv2d(ngf, out_channels, 3, 1, 1, bias=False)]

        # if activation == 'tanh':
        #     layers += [nn.Tanh()]
        # elif activation == 'sigmoid':
        #     layers += [nn.Sigmoid()]

        self.layers = nn.Sequential(*layers)

    def forward(self, x, z):
        # x is (batch, num_coords, 2)
        # z is (batch, latent_dim)

        if len(x.size()) < 3:
            x = x.unsqueeze(0)
        b = x.size(0)
        n = x.size(1)
        x = x.view(b*n, -1)
        if self.expand_coords:
            x2 = x**2
            xx = x[:,0]*x[:,1]
            x = torch.cat([x, x2, xx.unsqueeze(1)], 1)

        h_x = self.coord_linear(x)
        h_x = h_x.view(b, n, -1)

        h_z = 0
        if hasattr(self, 'latent_linear'):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            h_z = self.latent_linear(z)
            h_z = h_z.unsqueeze(1)

        h_bi = 0
        if hasattr(self, 'bilinear'):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            z = z.unsqueeze(1) # broadcast over coordinates
            x = x.view(b, n, -1)
            z = z.expand(b, x.size(1), z.size(2)).contiguous()
            h_bi = self.bilinear(x, z)

        h = h_x + h_z + h_bi # (batch, num_coords, hidden_dim)
        # h = h.view(b*n, -1)
        h = h.permute(0, 2, 1).reshape(b, h_x.shape[-1], *self.image_size)
        y = self.layers(h) # (batch*num_coords, nout)

        # if self.softplus: # only apply softplus to first output
        #     y = F.softplus(y)
        y = F.sigmoid(y)

        return y

class linearSpatialDecoder(nn.Module):
    def __init__(self, feat_dim, out_channels, hidden_dim, ngf, num_layers=1, activation=nn.Tanh
                , softplus=True, resid=False, expand_coords=False, bilinear=False):
        super(linearSpatialDecoder, self).__init__()

        self.image_size = [64, 64]
        self.softplus = softplus
        self.expand_coords = expand_coords
        self.out_channels = out_channels

        in_dim = 2
        if expand_coords:
            in_dim = 5 # include squares of coordinates as inputs

        self.coord_linear = nn.Linear(in_dim, hidden_dim)
        self.feat_dim = feat_dim
        if feat_dim > 0:
            self.latent_linear = nn.Linear(feat_dim, hidden_dim, bias=False)

        if feat_dim > 0 and bilinear: # include bilinear layer on latent and coordinates
            self.bilinear = nn.Bilinear(in_dim, feat_dim, hidden_dim, bias=False)

        layers = [activation()]
        for _ in range(1,num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim,hidden_dim))
                layers.append(activation())
        layers.append(nn.Linear(hidden_dim, out_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, z):
        # x is (batch, num_coords, 2)
        # z is (batch, latent_dim)

        if len(x.size()) < 3:
            x = x.unsqueeze(0)
        b = x.size(0)
        n = x.size(1)
        x = x.view(b*n, -1)
        if self.expand_coords:
            x2 = x**2
            xx = x[:,0]*x[:,1]
            x = torch.cat([x, x2, xx.unsqueeze(1)], 1)

        h_x = self.coord_linear(x)
        h_x = h_x.view(b, n, -1)

        h_z = 0
        if hasattr(self, 'latent_linear'):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            h_z = self.latent_linear(z)
            h_z = h_z.unsqueeze(1)

        h_bi = 0
        if hasattr(self, 'bilinear'):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            z = z.unsqueeze(1) # broadcast over coordinates
            x = x.view(b, n, -1)
            z = z.expand(b, x.size(1), z.size(2)).contiguous()
            h_bi = self.bilinear(x, z)

        h = h_x + h_z + h_bi # (batch, num_coords, hidden_dim)
        h = h.view(b*n, -1)

        y = self.layers(h) # (batch*num_coords, nout)
        y = y.view(b, n, -1)

        # if self.softplus: # only apply softplus to first output
        #     y = torch.cat([F.softplus(y[:,:,:1]), y[:,:,1:]], 2)
        # y = F.sigmoid(y)

        y = y.permute(0, 2, 1).reshape(b, self.out_channels, *self.image_size)
        return y

# class SpatialGenerator(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, n_out=1, num_layers=1, activation=nn.Tanh
#                 , softplus=False, resid=False, expand_coords=False, bilinear=False):
#         super(SpatialGenerator, self).__init__()
#
#         self.softplus = softplus
#         self.expand_coords = expand_coords
#
#         in_dim = 2
#         if expand_coords:
#             in_dim = 5 # include squares of coordinates as inputs
#
#         self.coord_linear = nn.Linear(in_dim, hidden_dim)
#         self.latent_dim = latent_dim
#         if latent_dim > 0:
#             self.latent_linear = nn.Linear(latent_dim, hidden_dim, bias=False)
#
#         if latent_dim > 0 and bilinear: # include bilinear layer on latent and coordinates
#             self.bilinear = nn.Bilinear(in_dim, latent_dim, hidden_dim, bias=False)
#
#         layers = [activation()]
#         for _ in range(1,num_layers):
#             if resid:
#                 layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
#             else:
#                 layers.append(nn.Linear(hidden_dim,hidden_dim))
#                 layers.append(activation())
#         layers.append(nn.Linear(hidden_dim, n_out))
#
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x, z):
#         # x is (batch, num_coords, 2)
#         # z is (batch, latent_dim)
#
#         if len(x.size()) < 3:
#             x = x.unsqueeze(0)
#         b = x.size(0)
#         n = x.size(1)
#         x = x.view(b*n, -1)
#         if self.expand_coords:
#             x2 = x**2
#             xx = x[:,0]*x[:,1]
#             x = torch.cat([x, x2, xx.unsqueeze(1)], 1)
#
#         h_x = self.coord_linear(x)
#         h_x = h_x.view(b, n, -1)
#
#         h_z = 0
#         if hasattr(self, 'latent_linear'):
#             if len(z.size()) < 2:
#                 z = z.unsqueeze(0)
#             h_z = self.latent_linear(z)
#             h_z = h_z.unsqueeze(1)
#
#         h_bi = 0
#         if hasattr(self, 'bilinear'):
#             if len(z.size()) < 2:
#                 z = z.unsqueeze(0)
#             z = z.unsqueeze(1) # broadcast over coordinates
#             x = x.view(b, n, -1)
#             z = z.expand(b, x.size(1), z.size(2)).contiguous()
#             h_bi = self.bilinear(x, z)
#
#         h = h_x + h_z + h_bi # (batch, num_coords, hidden_dim)
#         h = h.view(b*n, -1)
#
#         y = self.layers(h) # (batch*num_coords, nout)
#         y = y.view(b, n, -1)
#
#         if self.softplus: # only apply softplus to first output
#             y = torch.cat([F.softplus(y[:,:,:1]), y[:,:,1:]], 2)
#
#         return y