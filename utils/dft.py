import torch
import math
import torch.nn.functional as F

cos = lambda i, t, T=1.0: torch.cos(t*i*pi/T)
sin = lambda i, t, T=1.0: torch.sin(t*(i+1)*pi/T)
pi = math.pi

N = 504 * 378
w = torch.randn((N, 24*4)).cuda()           # the output of MLP
t = torch.randn((N, 64)).cuda()             # uniform t in test
d = torch.randn((N, 3)).cuda()              # ray direction
T = [(cos(i, t), sin(i, t))[int(i%2)].unsqueeze(-1) for i in range(24)]
T = torch.cat(T, dim=-1).cuda()             # pre-computed Trigonotic values
T = T.unsqueeze(1)                          # [N, 1, 64, 24]
ones = torch.ones((N, 1)).cuda()            # [N, 1]

def render_one_image(w):
    # compute the sigma and RGB of each point
    w = w.view(N, 4, 1, 24)                     # [N, 4, 1, 24]
    o = (w * T).sum(-1)                         # [N, 4, 64]
    sigma = o[:, 0,  :]
    rgbss = o[:, 1:, :]
    rgb = torch.sigmoid(rgbss).permute(0, 2, 1)

    # the volume rendering process as in NeRF
    sigma2alpha = lambda sigma, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(sigma)*dists)
    alpha = sigma2alpha(sigma, 1/64)

    weights = alpha * torch.cumprod(torch.cat([ones, 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N, 3]

    return rgb_map

out = render_one_image(w)
print(out.shape)