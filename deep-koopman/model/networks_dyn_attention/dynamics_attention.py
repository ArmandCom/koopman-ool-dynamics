from torch import nn
import torch


class DynamicsAttention(nn.Module):
    def __init__(self, n_objects, in_dim, dim, eps = 1e-8, hidden_dim = 64):
        super().__init__()
        self.n_objects = n_objects
        self.eps = eps
        self.scale = dim ** -0.5

        self.out_dim = dim
        self.gamma = nn.Parameter(torch.zeros(1))

        # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        # self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.to_k = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.to_v = nn.Conv2d(in_channels=in_dim, out_channels=dim, kernel_size=1)

        # self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=hidden_dim, kernel_size=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1),
        )

        # self.norm_input  = nn.LayerNorm(dim)
        # self.norm_slots  = nn.LayerNorm(dim)
        # self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, dyn_feat_inputs, ini_slots = None, num_slots = None):
        # b, T, n, h, w = inputs.shape
        # d = self.out_dim
        # inputs = inputs.reshape(b * T, n, h*w) # TODO: Check how layernorm affects batch
        # n_s = num_slots if num_slots is not None else self.num_slots
        #
        # if ini_slots is None:
        #     mu = self.slots_mu.expand(b, n_s, -1)
        #     sigma = self.slots_sigma.expand(b, n_s, -1)
        #     slots = torch.normal(mu, sigma)
        # else:
        #     slots = ini_slots
        #
        # # inputs = self.norm_input(inputs)
        # k, v = self.to_k(inputs).reshape(b, T, n, d), \
        #        self.to_v(inputs).reshape(b, T, n, d)
        #
        # slots_per_frame = []
        # for t in range(T):
        #
        #     slots_prev = torch.zeros_like(slots)
        #     for i in range(self.iters):
        #
        #         # slots = self.norm_slots(slots)
        #         q = self.to_q(slots)
        #
        #         k_t, v_t = k[:, t], v[:, t]
        #
        #         dots = torch.einsum('bid,bjd->bij', q, k_t) * self.scale
        #         attn = dots.softmax(dim=1) + self.eps
        #         attn = attn / attn.sum(dim=-1, keepdim=True)
        #
        #         updates = torch.einsum('bjd,bij->bid', v_t, attn)
        #
        #         slots = updates.reshape(-1, d)
        #         # slots = self.gru(
        #         #     updates.reshape(-1, d),
        #         #     slots_prev.reshape(-1, d)
        #         # )
        #
        #         slots = slots.reshape(b, -1, d)
        #
        #         slots = self.mlp(slots) + slots
        #         # slots = self.mlp(self.norm_pre_ff(slots)) #+ slots
        #         slots_prev = slots
        #
        #     slots_per_frame.append(slots)
        #
        # slots = torch.stack(slots_per_frame, dim=1)#.reshape(b, T, n, h, w)
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        bs, T, ch, w, h = inputs.size()
        bs2, n_obj, T2, ch2, w2, h2 = dyn_feat_inputs.size()
        bsTO = bs * n_obj * T
        inputs = inputs[:, None].repeat(1, n_obj, 1, 1, 1, 1).reshape(bsTO, ch, w, h)
        dyn_feat_inputs = dyn_feat_inputs.reshape(bsTO, ch, w, h)

        # Query, Key
        q = self.to_q(dyn_feat_inputs).view(bsTO, -1, w * h).permute(0, 2, 1)  # B X C X (W*H)
        k = self.to_k(inputs).view(bsTO, -1, w * h)  # B X C x (W*H)

        # Compute energy
        energy = torch.bmm(q, k) # * self.scale # transpose check, Why scale?

        # Original slot attention
        # energy_obj = energy.reshape(bs, n_obj, T, *energy.shape[1:])
        # attn_obj = energy_obj.softmax(dim=1) + self.eps

        # Self-attention
        attn_frame = energy.softmax(dim=-1) # + self.eps

        # We multiply by the union of both objects, as we don't want to amplify the 0ed regions.
        # attn = attn_obj * attn_frame.sum(dim=1, keepdim=True)
        # attn = attn / attn.sum(dim=-1, keepdim=True)

        attn = attn_frame

        # Get value and output
        v = self.to_v(inputs).reshape(bsTO, -1, w * h)  # B X C X N
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(bsTO, -1, w, h)

        # out = self.mlp(out) + out
        # out = self.gamma * out + inputs

        return out, attn