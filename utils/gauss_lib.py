import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from utils.fourier_triangle import iDFT

def minmax_norm(v, near, far, dim=-1):
    v_min, v_max = v.min(dim, True)[0], v.max(dim, True)[0]
    new_min, new_max = near, far

    v_new = (v-v_min)/(v_max-v_min) * (new_max-new_min) + new_min

    return v_new

def gauss_old(idx, raw, t, num_gau, near, far, epoch):
    """
    raw: [N_rays, 8x12], the guass parameters
    idx: int, the indication of rgb or sigma gauss
    t:   [N_sample], the sampled points 
    """
    num_ray = raw.shape[0]
    num_pts = t.shape[-1]
    num_gau = num_gau
    ratio = 1.0#(epoch / 2e3) + 1 if epoch < 2e5 else 100

    raw = raw.view(raw.shape[0], num_gau, -1).unsqueeze(1).repeat(1, num_pts, 1, 1) # [N_rays, N_sample, num_gau, 12=4x3]
    t = (t - near) / (far - near)
    t = t.unsqueeze(-1).repeat(1, 1, num_gau) # [N_sample, num_gau]
    # norm_mu, norm_dev = (far+near)/2.0, (far-near)/2.0

    # mu = raw[..., 3*idx+0]
    mu = torch.sigmoid(raw[..., 3*idx+0])                                           # [N_rays, N_sample, num_gau]
    # mu = 2.*(mu - 0.5)*norm_dev + norm_mu                                         

    # dev = raw[..., 3*idx+1]
    dev = torch.sigmoid(raw[..., 3*idx+1])                                          # [N_rays, N_sample, num_gau]
    # dev = (dev * norm_dev) / ratio                                                  

    inexp = -(t-mu)**2 / (2*dev**2+1e-6)                                            # [N_rays, N_sample, num_gau]
    outexp = (2*math.pi)**0.5 * dev                                                 # [N_rays, N_sample, num_gau]
    exp = 1 / (outexp+1e-6) * torch.exp(inexp)                                      # [N_rays, N_sample, num_gau]

    exp = exp.view(-1, num_gau).unsqueeze(2)                                        # [N_rays x N_sample, num_gau, 1]

    # if idx == 0:
    #     phi = F.relu(raw[..., 3*idx+2])                                             # [N_rays, N_sample, num_gau]
    # else:
    #     phi = F.normalize(torch.abs(raw[..., 3*idx+2]), p=1, dim=-1)                # [N_rays, N_sample, num_gau]
    phi = raw[..., 3*idx+2]
    phi = phi.view(-1, num_gau).unsqueeze(1)                                        # [N_rays x N_sample, 1, num_gau]
    res = phi.bmm(exp).squeeze().view(num_ray, num_pts)                             # [N_rays, N_sample]

    return res # [N_rays, N_sample]


def gauss_zvals(idx, raw, t, num_gau, near, far, epoch):
    """
    raw: [N_rays, 8x12], the guass parameters
    idx: int, the indication of rgb or sigma gauss
    t:   [N_sample], the sampled points 
    """
    num_ray = raw.shape[0]
    num_pts = t.shape[-1]
    num_gau = num_gau
    ratio = 1.0#(epoch / 2e3) + 1 if epoch < 2e5 else 100

    raw = raw.view(raw.shape[0], num_gau, -1).unsqueeze(1)                          # [N_rays, 1, num_gau, 12=4x3]
    t = (t - near) / (far - near)
    t = t.unsqueeze(-1)#.repeat(1, 1, num_gau)                                      # [N_rays, N_sample, 1]
    # norm_mu, norm_dev = (far+near)/2.0, (far-near)/2.0

    # mu = raw[..., 3*idx+0]
    mu = torch.sigmoid(raw[..., 3*idx+0])                                           # [N_rays, 1, num_gau]
    # mu = 2.*(mu - 0.5)*norm_dev + norm_mu                                         

    # dev = raw[..., 3*idx+1]
    dev = torch.sigmoid(raw[..., 3*idx+1])                                          # [N_rays, 1, num_gau]
    # dev = (dev * norm_dev) / ratio                                                  

    inexp = -(t-mu)**2 / (2*dev**2+1e-6)                                            # [N_rays, N_sample, num_gau]
    outexp = (2*math.pi)**0.5 * dev                                                 # [N_rays, N_sample, num_gau]
    exp = 1 / (outexp+1e-6) * torch.exp(inexp)                                      # [N_rays, N_sample, num_gau]

    phi = raw[..., 3*idx+2]
    phi = phi.view(-1, num_gau).unsqueeze(1)                                        # [N_rays, 1, num_gau]
    res = (phi * exp).sum(-1)                                                       # [N_rays, N_sample]

    return res # [N_rays, N_sample]


def gauss_dft(idx, raw, t, num_gau, near, far, epoch):
    """
    raw: [N_rays, N_freq x 12], the guass parameters, here num_gau == N_freq
    idx: int, the indication of rgb or sigma gauss
    t:   [N_sample], the sampled points 
    """
    raw = raw.view(raw.shape[0], num_gau, -1)                                       # [N_rays, num_gau, 4]
    phi = raw[..., idx].unsqueeze(1)                                                # [N_rays, 1, num_gau]
    
    t = (t - near) / (far - near)                                                   # [N_rays, N_sample]
    idft = iDFT(t, num_gau)                                                         # [N_rays, N_sample, num_gau]
    
    res = (phi * idft).sum(-1)                                                      # [N_rays, N_sample]

    return res # [N_rays, N_sample]


def gauss_uncert(idx, raw, t, num_gau, near, far, epoch):
    """
    raw: [N_rays, 8x12], the guass parameters
    idx: int, the indication of rgb or sigma gauss
    t:   [N_sample], the sampled points ### no use
    """
    num_ray = raw.shape[0]
    num_gau = num_gau
    ratio = 1.0#(epoch / 2e3) + 1 if epoch < 2e5 else 100

    raw = raw.view(raw.shape[0], num_gau, -1).unsqueeze(1)                          # [N_rays, 1, num_gau, 12=4x3]

    mu = torch.sigmoid(raw[..., 3*idx+0])                                           # [N_rays, 1, num_gau]
    dev = torch.sigmoid(raw[..., 3*idx+1])                                          # [N_rays, 1, num_gau]

    noise = torch.rand(dev.shape) - 0.5
    t = (mu + dev * noise).squeeze(1).unsqueeze(-1)                                 # [N_rays, N_sample, 1]

    inexp = -(t-mu)**2 / (2*dev**2+1e-6)                                            # [N_rays, N_sample, num_gau]
    outexp = (2*math.pi)**0.5 * dev                                                 # [N_rays, N_sample, num_gau]
    exp = 1 / (outexp+1e-6) * torch.exp(inexp)                                      # [N_rays, N_sample, num_gau]

    phi = raw[..., 3*idx+2]
    phi = phi.view(-1, num_gau).unsqueeze(1)                                        # [N_rays, 1, num_gau]
    res = (phi * exp).sum(-1)                                                       # [N_rays, N_sample]

    return res # [N_rays, N_sample]


gauss_dict = {'zvals': gauss_zvals, 'uncert': gauss_uncert, 'dft': gauss_dft}


def MYraw2outputs(args, epoch, raw, z_vals, near, far, rays_d, N_gauss, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    t = z_vals
    num_gau = N_gauss
    gauss = gauss_dict[args.gauss_type]
    sigma_t = gauss(0, raw, t, num_gau, near, far, epoch)
    r_t = gauss(1, raw, t, num_gau, near, far, epoch).unsqueeze(-1)
    g_t = gauss(2, raw, t, num_gau, near, far, epoch).unsqueeze(-1)
    b_t = gauss(3, raw, t, num_gau, near, far, epoch).unsqueeze(-1)
    rgb = torch.cat([r_t, g_t, b_t], dim=-1)  # [N_rays, N_samples, 3]
    rgb = torch.sigmoid(rgb)
    # print(sigma_t.max(), sigma_t.min())

    ##################### topk ######################
    # sigma_thresh = sigma_t.topk(num_gau, dim=1) # [N_rays, num_gau]
    # sigma_thresh = sigma_thresh.values[:, -1].unsqueeze(-1)
    # index_suppress = sigma_t < sigma_thresh
    # sigma_t[index_suppress] = 0
    ##################### topk ######################
    
    sigma2alpha = lambda sigma, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(sigma)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(sigma_t.shape) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(sigma_t.shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # noise = 1e-3
    alpha = sigma2alpha(sigma_t + noise, dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, sigma_t, rgb


def get_sigma(epoch, raw, z_vals, near, far, rays_d, N_gauss, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    t = z_vals
    num_gau = N_gauss
    gauss = gauss_zvals
    sigma_t = gauss(0, raw, t, num_gau, near, far, epoch)
    r_t = gauss(1, raw, t, num_gau, near, far, epoch).unsqueeze(-1)
    g_t = gauss(2, raw, t, num_gau, near, far, epoch).unsqueeze(-1)
    b_t = gauss(3, raw, t, num_gau, near, far, epoch).unsqueeze(-1)
    rgb = torch.cat([r_t, g_t, b_t], dim=-1)  # [N_rays, N_samples, 3]
    rgb = torch.sigmoid(rgb)

    return sigma_t, rgb


def get_dev(raw, N_gauss):
    
    idx = 0
    raw = raw.view(raw.shape[0], N_gauss, -1)  # [N_rays, num_gau, 12=4x3]
    # mu = torch.sigmoid(raw[..., 3*idx+0])    # [N_rays, num_gau]
    # phi = raw[..., 3*idx+2]                  # [N_rays, num_gau]
    dev = torch.sigmoid(raw[..., 3*idx+1])     # [N_rays, num_gau]
    # dev = torch.flatten(dev)                   # [N_rays * num_gau]

    return dev