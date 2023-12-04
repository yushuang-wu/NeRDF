import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def get_hwf(hwf):
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    return hwf

def get_newK(hwf):
    H, W, focal = hwf
    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    return K


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NewNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, num_Gaussian=8, skips=[4]):
        """ 
        """
        super(MYNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.num_Gau = num_Gaussian
        self.skips = skips
        
        self.o_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.d_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.output_linear = nn.Linear(W, num_Gaussian*3*4)

    def forward(self, o, d):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        outputs = self.output_linear(h)

        return outputs

# Model
class MYNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, num_Gaussian=8, skips=[4]):
        """ 
        """
        super(MYNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.num_Gau = num_Gaussian
        self.skips = skips
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(2*input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + 2*input_ch, W) for i in range(D-1)])

        self.output_linear = nn.Linear(W, num_Gaussian*3*4)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        outputs = self.output_linear(h)

        return outputs

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
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
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # noise = 1e-3
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, raw[...,3], rgb


def minmax_norm(v, near, far, dim=-1):
    v_min, v_max = v.min(dim, True)[0], v.max(dim, True)[0]
    new_min, new_max = near, far

    v_new = (v-v_min)/(v_max-v_min) * (new_max-new_min) + new_min

    return v_new

def gauss(idx, raw, t, num_gau, near, far):
    """
    raw: [N_rays, 8x12], the guass parameters
    idx: int, the indication of rgb or sigma gauss
    t:   [N_sample], the sampled points 
    """
    num_ray = raw.shape[0]
    num_pts = t.shape[0]
    num_gau = num_gau

    raw = raw.view(raw.shape[0], num_gau, -1).unsqueeze(1).repeat(1, num_pts, 1, 1) # [N_rays, N_sample, num_gau, 12=4x3]
    t = t.unsqueeze(1).repeat(1, num_gau)                                           # [N_sample, num_gau]
    norm_mu, norm_dev = (far+near)/2.0, (far-near)/2.0

    mu_normalized = F.normalize(raw[..., 3*idx+0], p=2, dim=-1)                     # [N_rays, N_sample, num_gau]
    mu = 2.*(torch.abs(mu_normalized) - 0.5)*norm_dev + norm_mu                     # [N_rays, N_sample, num_gau]
    # mu = minmax_norm(raw[..., 3*idx+0], near, far)

    dev_normalized = F.normalize(raw[..., 3*idx+1], p=2, dim=-1)                    # [N_rays, N_sample, num_gau]
    dev = torch.abs(dev_normalized * norm_dev)                                      # [N_rays, N_sample, num_gau]
    # dev = torch.ones_like(mu) * 0.01

    inexp = -(t-mu)**2 / (2*dev**2+1e-10)                                           # [N_rays, N_sample, num_gau]
    outexp = (2*math.pi)**0.5 * dev                                                 # [N_rays, N_sample, num_gau]
    exp = 1 / (outexp+1e-6) * torch.exp(inexp)                                      # [N_rays, N_sample, num_gau]

    exp = exp.view(-1, num_gau).unsqueeze(2)                                        # [N_rays x N_sample, num_gau, 1]

    phi = F.relu(raw[..., 3*idx+2])                                                 # [N_rays, N_sample, num_gau]
    # phi = F.normalize(phi, p=1, dim=-1)                                           # [N_rays, N_sample, num_gau]
    phi = phi.view(-1, num_gau).unsqueeze(1)                                        # [N_rays x N_sample, 1, num_gau]
    res = phi.bmm(exp).squeeze().view(num_ray, num_pts)                             # [N_rays, N_sample]

    return res # [N_rays, N_sample]

def gauss_single(idx, raw, t, num_gau, near, far):
    """
    raw: [N_rays, 8x12], the guass parameters
    idx: int, the indication of rgb or sigma gauss
    t:   [N_rays], one point per ray
    """
    num_ray = raw.shape[0]
    num_pts = t.shape[0]
    num_gau = num_gau

    raw = raw.view(raw.shape[0], num_gau, -1)                                       # [N_rays, num_gau, 12=4x3]
    t = t.unsqueeze(1).repeat(1, num_gau)                                           # [N_rays, num_gau]
    norm_mu, norm_dev = (far+near)/2.0, (far-near)/2.0

    mu_normalized = F.normalize(raw[..., 3*idx+0], p=2, dim=-1)                     # [N_rays, num_gau]
    mu = 2.*(torch.abs(mu_normalized) - 0.5)*norm_dev + norm_mu                     # [N_rays, num_gau]
    # mu = minmax_norm(raw[..., 3*idx+0], near, far)

    dev_normalized = F.normalize(raw[..., 3*idx+1], p=2, dim=-1)                    # [N_rays, num_gau]
    dev = torch.abs(dev_normalized * norm_dev)                                      # [N_rays, num_gau]
    # dev = torch.ones_like(mu) * 0.01

    inexp = -(t-mu)**2 / (2*dev**2+1e-10)                                           # [N_rays, num_gau]
    outexp = (2*math.pi)**0.5 * dev                                                 # [N_rays, num_gau]
    exp = 1 / (outexp+1e-6) * torch.exp(inexp)                                      # [N_rays, num_gau]

    phi = F.relu(raw[..., 3*idx+2])                                                 # [N_rays, num_gau]
    # phi = F.normalize(phi, p=1, dim=-1)                                           # [N_rays, num_gau]
    res = (phi * exp).sum(-1)                                                       # [N_rays]

    return res # [N_rays]


def gauss_mudev(idx, raw, num_gau, near, far):
    """
    raw: [N_rays, 8x12], the guass parameters
    idx: int, the indication of rgb or sigma gauss
    """
    num_ray = raw.shape[0]
    num_gau = num_gau

    raw = raw.view(raw.shape[0], num_gau, -1)                                       # [N_rays, num_gau, 12=4x3]
    norm_mu, norm_dev = (far+near)/2.0, (far-near)/2.0

    mu_normalized = F.normalize(raw[..., 3*idx+0], p=2, dim=-1)                     # [N_rays, num_gau]
    mu = 2.*(torch.abs(mu_normalized) - 0.5)*norm_dev + norm_mu                     # [N_rays, num_gau]
    # mu = minmax_norm(raw[..., 3*idx+0], near, far)

    dev_normalized = F.normalize(raw[..., 3*idx+1], p=2, dim=-1)                    # [N_rays, num_gau]
    dev = torch.abs(dev_normalized * norm_dev)                                      # [N_rays, num_gau]
    # dev = torch.ones_like(mu) * 0.01

    return mu, dev


def MYraw2outputs(raw, z_vals, near, far, rays_d, N_gauss, raw_noise_std=0, white_bkgd=False, pytest=False):
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
    sigma_t = gauss(0, raw, t, num_gau, near, far)
    r_t = gauss(1, raw, t, num_gau, near, far).unsqueeze(-1)
    g_t = gauss(2, raw, t, num_gau, near, far).unsqueeze(-1)
    b_t = gauss(3, raw, t, num_gau, near, far).unsqueeze(-1)
    rgb = torch.cat([r_t, g_t, b_t], dim=-1)  # [N_rays, N_samples, 3]
    rgb = torch.sigmoid(rgb - rgb.mean())
    # print(sigma_t.max(), sigma_t.min())
    
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

    noise = 1e-3
    alpha = sigma2alpha(sigma_t + noise, dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, sigma_t, rgb


def get_newview(raw_org, rays_o_add, rays_o, viewdirs, near, far, device, num_gau=8, num_sample=8, uniform_sampling=True):
    """
    raw: [N_rays, 8x12], the guass parameters
    idx: int, the indication of rgb or sigma gauss
    """
    num_ray = raw_org.shape[0]
    if not uniform_sampling:
        mu, dev = gauss_mudev(0, raw_org, num_gau, near, far)
        mu, dev = mu.unsqueeze(-1).repeat(1, 1, num_sample), dev.unsqueeze(-1).repeat(1, 1, num_sample)  # [N_rays, num_gau, num_sample]
        t_org = torch.randn(num_ray, num_gau, num_sample).to(device) - 0.5  # [N_rays, num_gau, num_sample]
        t_org = (mu + t_org).view(num_ray, -1)
        # t_org = (mu + t_org * dev).view(num_ray, -1)                    # [N_rays, num_gau * num_sample]
    else:
        t_vals = torch.linspace(0., 1., steps=128)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        t_org = torch.tensor(np.random.choice(z_vals.cpu(), (num_ray, num_gau*num_sample))).to(device)  # [N_rays, num_pts_add]

    pts_add = rays_o.unsqueeze(1).repeat(1, num_gau*num_sample, 1) + \
              viewdirs.unsqueeze(1).repeat(1, num_gau*num_sample, 1) * \
              t_org.unsqueeze(-1)  # [N_rays, num_gau * num_sample, 3]

    rays_d_add = pts_add - rays_o_add
    viewdirs_add = rays_d_add / torch.norm(rays_d_add, dim=-1, keepdim=True)
    viewdirs_add = viewdirs_add.view(-1, 3).float().to(device)
    t_org = t_org.view(-1).float().to(device)
    t_add = (rays_d_add.view(-1, 3) / (viewdirs_add + 1e-10))[:, 0].float().to(device)
    mask = (t_add > near) & (t_add < far) & (t_org > near) & (t_org < far)  # [N_rays * num_gau * num_sample, 3]

    return viewdirs_add, t_org, t_add, mask


def compute_constrain(raw_org, raw_add, t_org, t_add, mask, near, far, device, num_gau=8, num_sample=8):
    num_pts_add = int(raw_add.shape[0]/raw_org.shape[0])
    raw_org = raw_org.unsqueeze(1).repeat(1, num_pts_add, 1).view(-1, raw_org.shape[-1])                    # [N_rays * num_pts_add, 96]
    sigma_org = gauss_single(0, raw_org, t_org, num_gau, near, far).to(device)                              # [N_rays * num_pts_add]
    sigma_add = gauss_single(0, raw_add, t_add, num_gau, near, far).to(device)                              # [N_rays * num_pts_add]
    # print(sigma_org.max(), sigma_org.min())
    constrain = F.l1_loss(sigma_org, sigma_add, reduce=None) * mask
    constrain = constrain.mean(-1)

    return constrain

def intersection_constrain(rays_o, viewdirs):
    viewdirs_a, viewdirs_b = torch.broadcast_tensors(viewdirs.unsqueeze(1), viewdirs.unsqueeze(0))
    cross_product = torch.cross(viewdirs_a, viewdirs_b)
    cross_origins = rays_o.unsqueeze(1) - rays_o.unsqueeze(0)

    distance = cross_origins.mul(cross_product).sum(-1)
    distance[torch.where(cross_product.sum(-1) > 0)] = 100.0

    indices = (distance < 0.1).float().nonzero()
    index_0, index_1 = indices[:, 0], indices[:, 1]

    return constrain

