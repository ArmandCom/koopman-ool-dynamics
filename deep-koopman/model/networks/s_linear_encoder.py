import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.Tanh):
        super(ResidLinear, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)


class LinearEncoder(nn.Module):
    def __init__(self, n, latent_dim, hidden_dim, num_layers=1, activation=nn.Tanh, resid=False):
        super(LinearEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.n = n

        layers = [nn.Linear(n, hidden_dim),
                  activation(),
                 ]
        for _ in range(1, num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())

        layers.append(nn.Linear(hidden_dim, 2*latent_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x is (batch,num_coords)
        if len(x.shape)==4:
            x = x.reshape(x.shape[0], -1)
        return self.layers(x)