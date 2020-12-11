from torch import nn
import torch


class TemporalSlotAttention(nn.Module):
    def __init__(self, num_slots, in_dim, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.out_dim = dim

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(in_dim, dim)
        self.to_v = nn.Linear(in_dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, ini_slots = None, num_slots = None):
        b, T, n, h, w = inputs.shape
        d = self.out_dim
        inputs = inputs.reshape(b * T, n, h*w) # TODO: Check how layernorm affects batch
        n_s = num_slots if num_slots is not None else self.num_slots

        if ini_slots is None:
            mu = self.slots_mu.expand(b, n_s, -1)
            sigma = self.slots_sigma.expand(b, n_s, -1)
            slots = torch.normal(mu, sigma)
        else:
            slots = ini_slots

        # inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs).reshape(b, T, n, d), \
               self.to_v(inputs).reshape(b, T, n, d)

        slots_per_frame = []
        for t in range(T):

            slots_prev = torch.zeros_like(slots)
            for i in range(self.iters):

                # slots = self.norm_slots(slots)
                q = self.to_q(slots)

                k_t, v_t = k[:, t], v[:, t]

                dots = torch.einsum('bid,bjd->bij', q, k_t) * self.scale
                attn = dots.softmax(dim=1) + self.eps
                attn = attn / attn.sum(dim=-1, keepdim=True)

                updates = torch.einsum('bjd,bij->bid', v_t, attn)

                slots = updates.reshape(-1, d)
                # slots = self.gru(
                #     updates.reshape(-1, d),
                #     slots_prev.reshape(-1, d)
                # )

                slots = slots.reshape(b, -1, d)

                slots = self.mlp(slots) + slots
                # slots = self.mlp(self.norm_pre_ff(slots)) #+ slots
                slots_prev = slots

            slots_per_frame.append(slots)

        slots = torch.stack(slots_per_frame, dim=1)#.reshape(b, T, n, h, w)
        return slots

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots
