import torch
from utils.spherical_harmonics import embed_SH

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network1(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


# def run_network2(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
#     """Prepares inputs and applies network 'fn'.
#     """
#     num_rays, num_pts, dim = inputs.shape
#     inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
#     embedded = embed_fn(inputs_flat)

#     if viewdirs is not None:
#         input_dirs = viewdirs[:,None].expand(inputs.shape)
#         input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
#         embedded_dirs = embeddirs_fn(input_dirs_flat)
#         embedded = torch.cat([embedded, embedded_dirs], -1)

#     embedded = torch.reshape(embedded, [num_rays, num_pts, -1])
#     embedded = torch.reshape(embedded, [num_rays, -1])

#     outputs = batchify(fn, netchunk)(embedded)
#     return outputs


def run_network2_ptsFreq(pts, rays_o, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    L = 10
    x = pts 
    weights = 2**torch.linspace(0, L - 1, steps=L).cuda()
    y = x[..., None] * weights                             # [n_ray, dim_pts, 1] * [L] -> [n_ray, dim_pts, L]
    y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)    # [n_ray, dim_pts, 2L]
    y = torch.cat([y, x.unsqueeze(dim=-1)], dim=-1)        # [n_ray, dim_pts, 2L+1]
    y = y.view(y.shape[0], -1)                             # [n_ray, dim_pts*(2L+1)], example: 48*21=1008
    
    outputs = batchify(fn, netchunk)(y)
    return outputs


# pts and rayso use frequency encoding, dirs use spherical harmonics
def run_network2_optsFreq_dSH(pts, rays_o, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    L = 10
    x = pts 
    weights = 2**torch.linspace(0, L - 1, steps=L).cuda()
    y = x[..., None] * weights                             # [n_ray, dim_pts, 1] * [L] -> [n_ray, dim_pts, L]
    y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)    # [n_ray, dim_pts, 2L]
    y = torch.cat([y, x.unsqueeze(dim=-1)], dim=-1)        # [n_ray, dim_pts, 2L+1]
    y = y.view(y.shape[0], -1)                             # [n_ray, dim_pts*(2L+1)], example: 48*21=1008
    
    num_rays = rays_o.shape[0]

    rayo_flat = torch.reshape(rays_o, [-1, rays_o.shape[-1]])
    embedded_o = embed_fn(rayo_flat)

    dirs_flat = torch.reshape(viewdirs, [-1, viewdirs.shape[-1]])
    embedded_d = embed_SH(dirs_flat)

    embedded = torch.cat([y, embedded_o, embedded_d], -1)
    # print(embedded.shape)
    
    outputs = batchify(fn, netchunk)(embedded)
    return outputs


def run_network2_odFreq(pts, rays_o, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    num_rays = rays_o.shape[0]
    inputs_flat = torch.reshape(rays_o, [-1, rays_o.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    embedded = torch.reshape(embedded, [num_rays, -1])

    # print(embedded.shape)
    outputs = batchify(fn, netchunk)(embedded)
    return outputs


def run_network2_ptsodFreq(pts, rays_o, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    num_rays = rays_o.shape[0]
    inputs_flat = torch.reshape(rays_o, [-1, rays_o.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    embedded = torch.reshape(embedded, [num_rays, -1])

    L = 10
    x = pts 
    weights = 2**torch.linspace(0, L - 1, steps=L).cuda()
    y = x[..., None] * weights                             # [n_ray, dim_pts, 1] * [L] -> [n_ray, dim_pts, L]
    y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)    # [n_ray, dim_pts, 2L]
    y = torch.cat([y, x.unsqueeze(dim=-1)], dim=-1)        # [n_ray, dim_pts, 2L+1]
    y = y.view(y.shape[0], -1)                             # [n_ray, dim_pts*(2L+1)], example: 48*21=1008
    
    embedded2 = torch.cat([embedded, y], -1)
    # print(embedded2.shape)
    outputs = batchify(fn, netchunk)(embedded2)
    return outputs


def batchify2(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs, depth4):
        return torch.cat([fn(inputs[i:i+chunk], depth4) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network2_d4(pts, rays_o, viewdirs, fn, depth4, embed_fn, embeddirs_fn, netchunk=1024*64):
    L = 10
    x = pts 
    weights = 2**torch.linspace(0, L - 1, steps=L).cuda()
    y = x[..., None] * weights                             # [n_ray, dim_pts, 1] * [L] -> [n_ray, dim_pts, L]
    y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)    # [n_ray, dim_pts, 2L]
    y = torch.cat([y, x.unsqueeze(dim=-1)], dim=-1)        # [n_ray, dim_pts, 2L+1]
    y = y.view(y.shape[0], -1)                             # [n_ray, dim_pts*(2L+1)], example: 48*21=1008
    
    outputs = batchify2(fn, netchunk)(y, depth4)
    return outputs


run_network2_dict = {'ptsFreq':      run_network2_ptsFreq, 
                     'optsFreq_dSH': run_network2_optsFreq_dSH, 
                     'odFreq':       run_network2_odFreq,
                     'odptsFreq':    run_network2_ptsodFreq, 
                     'depth4':       run_network2_d4}

input_ch_dict = {'ptsFreq':      1008, 
                 'optsFreq_dSH': 1135, 
                 'odFreq':       1008,
                 'odptsFreq':    1008, 
                 'depth4':       1008}
