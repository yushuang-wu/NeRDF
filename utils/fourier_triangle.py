import torch
import math

def iDFT(t, n2):
    '''
    t:    [N_rays, N_samples], (0, 1)
    idft: [N_rays, N_samples, n2], n2 is # of components (num_gau)
    '''
    pi = math.pi
    N_rays, N_samples = t.shape[0], t.shape[1]
    idft = torch.zeros([N_rays, n2])

    cos = lambda i, t, T=1.0: torch.cos(t*i*pi/T)
    sin = lambda i, t, T=1.0: torch.sin(t*(i+1)*pi/T)

    idft = [(cos(i, t), sin(i, t))[int(i%2)].unsqueeze(-1) for i in range(n2)]
    idft = torch.cat(idft, dim=-1)

    return idft
