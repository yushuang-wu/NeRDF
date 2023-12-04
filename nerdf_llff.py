import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from helpers import *
from utils.input_emb import *
from utils.model_lib import *
from utils.gauss_lib import *

from load_llff import *
from load_blender import load_blender_data

from option import config_parser
from functools import partial

import lpips as lpips_
from utils.ssim_torch import ssim as ssim_

from smilelogging.utils import Timer, LossLine, get_n_params_, get_n_flops_, AverageMeter, ProgressMeter

ssim = lambda img, ref: ssim_(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))
lpips = lpips_.LPIPS(net='alex').cuda()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

parser = config_parser()
args = parser.parse_args()

if args.gcr:
    print('Using GCR Training!')
    args.basedir = os.path.join(os.environ['AMLT_OUTPUT_DIR'], args.basedir)
    args.datadir = os.path.join('/mnt/default/', args.datadir)
    args.teacher_ckpt = os.path.join('/mnt/default/', args.teacher_ckpt)
    if not os.path.exists(args.basedir):
        os.makedirs(args.basedir)
else:
    print('Local Training!')


def batchify_rays(epoch, rays_flat, hwf, N_gauss, render_test, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(epoch, rays_flat[i:i+chunk], hwf, N_gauss, render_test, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret_new = {k : torch.cat(all_ret[k], 0) for k in all_ret if k != 'constrain'}
    all_ret_new['constrain'] = 0
    for i in all_ret['constrain']:
        all_ret_new['constrain'] += i
    return all_ret_new


def render(epoch, H, W, K, chunk=1024*32, N_gauss=4, 
           rays=None, c2w=None, ndc=True,
           near=0., far=1., render_test=False, 
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays_v2(H, W, K[0][0], c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays_v2(H, W, K[0][0], c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    hwf = [H, W, K]
    all_ret = batchify_rays(epoch, rays, hwf, N_gauss, render_test, chunk, **kwargs)
    if render_test:
        for k in all_ret:
            if k == 'constrain':
                continue
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = [
        'rgb_map1', 'disp_map1', 'acc_map1', 
        'rgb_map2', 'disp_map2', 'acc_map2', 
        'constrain'
        ]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(epoch, render_poses, hwf, chunk, N_gauss, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf
    K = get_newK(hwf)

    rgbs = []
    disps = []

    for i, c2w in enumerate(tqdm(render_poses)):
        rgb1, disp1, acc1, rgb2, disp2, acc2, _, _ = render(epoch, H, W, K, chunk=chunk, N_gauss=N_gauss, c2w=c2w[:3,:4], render_test=True, **render_kwargs)
        rgbs.append(rgb2.cpu().numpy())
        disps.append(disp2.cpu().numpy())
        if i==0:
            print(rgb2.shape, disp2.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    embed_fn_gs, input_ch_gs = get_embedder(args.multires_gs, args.i_embed)
    input_ch_views_gs = 0
    embeddirs_fn_gs = None
    if args.use_viewdirs:
        embeddirs_fn_gs, input_ch_views_gs = get_embedder(args.multires_views_gs, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4 * i for i in range(8) if i != 0]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    # grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        # grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network1(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    ##################################################################################################
    model_gs = model_gs_dict[args.model_gs_type]
    input_ch = input_ch_dict[args.embedding_type]
    output_ch = args.N_gauss*4 if args.model_gs_type == 'dft' else args.N_gauss*3*4
    gauss = model_gs(D=args.netdepth_gs, W=args.netwidth_gs,
                     input_ch=input_ch, output_ch=output_ch, skips=skips).to(device)
    # torch.save(gauss.state_dict(), 'utils/ckpt_example.tar')
    # print('SAVED!')
    grad_vars = list(gauss.parameters())

    run_network2 = run_network2_dict[args.embedding_type]
    network_query_gs = lambda pts, rays_o, viewdirs, network_fn : run_network2(pts, rays_o, viewdirs, network_fn,
                                                                 embed_fn=embed_fn_gs,
                                                                 embeddirs_fn=embeddirs_fn_gs,
                                                                 netchunk=args.netchunk)

    print('Model_gs Type: ', args.model_gs_type)
    print('Embedding Type: ', args.embedding_type)
    print('Gaussian Type: ', args.gauss_type)
    ##################################################################################################

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    print('Teacher ckpt', args.teacher_ckpt)
    if not args.render_test:
        ckpt1 = torch.load(args.teacher_ckpt)
        model.load_state_dict(ckpt1['network_fn_state_dict'])
        model_fine.load_state_dict(ckpt1['network_fine_state_dict'])
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        gauss.load_state_dict(ckpt['network_fn_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'network_query_gs' : network_query_gs,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'network_gs' : gauss,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    # print net parameter number and n_flops
    n_params = get_n_params_(gauss)
    dummy_input = torch.randn(1, gauss.input_ch).to(device)
    n_flops = get_n_flops_(gauss, input=dummy_input, count_adds=False)

    print(
        f'Model complexity per pixel: FLOPs {n_flops/1e6:.10f}M, Params {n_params/1e6:.10f}M'
    )

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def perturb_rays(z_vals, pytest):
    # get intervals between samples
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        t_rand = np.random.rand(*list(z_vals.shape))
        t_rand = torch.Tensor(t_rand)

    z_vals = lower + (upper - lower) * t_rand

    return z_vals

def render_rays(epoch, 
                ray_batch, 
                hwf, 
                N_gauss, 
                render_test,
                network_fn,
                network_gs,
                network_query_fn,
                network_query_gs,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    H, W, K = hwf
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0][0], bounds[...,1][1] # [-1,1]
    N_rays = rays_o.shape[0]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])
    if perturb > 0.:
        z_vals = perturb_rays(z_vals, pytest)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map1, disp_map1, acc_map1, weights1, depth_map1, sigma1, rgb_map1 = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map1, disp_map1, acc_map1
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights1[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map1, disp_map1, acc_map1, weights1, depth_map1, sigma1, rgb1 = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ######################################################################

    N_samples = 16
    t_vals2 = torch.linspace(0., 1., steps=N_samples)
    z_vals2_flat = near * (1.-t_vals2) + far * (t_vals2)
    z_vals2 = z_vals2_flat.expand([N_rays, N_samples])
    if perturb > 0.:
        z_vals2 = perturb_rays(z_vals2, pytest)
    pts2 = rays_o[...,None,:] + rays_d[...,None,:] * z_vals2[...,:,None] # [N_rays, 16, 3]

    
    if not render_test:
        Sample_pred = args.N_sample_train # train
    else:
        Sample_pred = args.N_sample_test # test
    t_vals_pred = torch.linspace(0., 1., steps=Sample_pred)
    z_vals_pred = near * (1.-t_vals_pred) + far * (t_vals_pred)
    z_vals_pred = z_vals_pred.expand([N_rays, Sample_pred])
    if perturb > 0.:
        z_vals_pred = perturb_rays(z_vals_pred, pytest)

    # torch.cuda.synchronize()
    # t_inp = time.time()
    raw_org2 = network_query_gs(pts2, rays_o, viewdirs, network_gs)  # [N_rays, 3]
    # torch.cuda.synchronize()
    # t_raw = time.time()
    rgb_map2, disp_map2, acc_map2, *_ = MYraw2outputs(args, epoch, raw_org2, z_vals_pred, near, far, rays_d, N_gauss, raw_noise_std, white_bkgd, pytest=pytest)
    # torch.cuda.synchronize()
    # t_rgb = time.time()
    # print('Forward time: ', t_raw - t_inp)
    # print('Render time:  ', t_rgb - t_raw)
    
    if render_test:
        constrain = torch.tensor(0)

    elif args.sigma_constrain or args.dev_regularization:
        constrain1, constrain2 = torch.tensor(0), torch.tensor(0)

        if args.sigma_constrain:
            sigma2, rgb2 = get_sigma(epoch, raw_org2, z_vals, near, far, rays_d, N_gauss, raw_noise_std, white_bkgd, pytest=pytest)

            ### sigma constrain ###
            # sigma1_norm = F.normalize(torch.clip(sigma1, min=0), dim=-1)
            # sigma2_norm = F.normalize(torch.clip(sigma2, min=0), dim=-1)
            sigma1_norm = F.normalize(sigma1, dim=-1)
            sigma2_norm = F.normalize(sigma2, dim=-1)
            sigma_const = F.mse_loss(sigma2_norm, sigma1_norm.detach())
            constrain1 = sigma_const

            # torch.save(sigma1_norm.detach().cpu(), f'{args.basedir}/{args.expname}/sigma_vis/sigma1.pth')
            # torch.save(sigma2_norm.detach().cpu(), f'{args.basedir}/{args.expname}/sigma_vis/sigma2.pth')
            # torch.save(z_vals.detach().cpu(), f'{args.basedir}/{args.expname}/sigma_vis/z_vals.pth')
            # print(sigma1_norm.shape, sigma2_norm.shape, z_vals.shape)
            # print('sigma saved!')
            # raise NotImplementedError
            
            ### rgb constrain ###
            # rgb_const = F.mse_loss(rgb2, rgb1.detach())
            # constrain1 = sigma_const + rgb_const

            ### sigma peak position ###
            # peak1 = sigma1.max(1)[1].float().detach()
            # peak2 = sigma2.max(1)[1].float()
            # constrain1 = F.l1_loss(peak2, peak1)

            ### sigma constrain KL divergence###
            # kl_loss = nn.KLDivLoss(reduction="batchmean")
            # # sigma1_norm = F.normalize(F.relu(sigma1), dim=-1)
            # # sigma2_norm = F.normalize(F.relu(sigma2), dim=-1)
            # sigma1_norm = F.softmax(minmax_norm(sigma1, 0., 1.), dim=-1) # [1024, 128]
            # sigma2_norm = F.softmax(minmax_norm(sigma2, 0., 1.), dim=-1)
            # sigma1_norm = F.softmax(F.normalize(sigma1, dim=-1)) # [1024, 128]
            # sigma2_norm = F.softmax(F.normalize(sigma2, dim=-1))
            # sigma_const = kl_loss(sigma2_norm, sigma1_norm.detach())
            # constrain1 = sigma_const

        if args.dev_regularization:
            dev = get_dev(raw_org2, N_gauss)
            constrain2 = torch.norm(dev, p=2) * 0.1

        constrain = constrain1 + constrain2

    else:
        constrain = torch.tensor(0)
    
    ret = {
        'rgb_map1' : rgb_map1, 'disp_map1' : disp_map1, 'acc_map1' : acc_map1, 
        'rgb_map2' : rgb_map2, 'disp_map2' : disp_map2, 'acc_map2' : acc_map2, 
        'constrain' : constrain
        }
    if retraw:
        ret['raw1'] = raw
        ret['raw2'] = raw_org2

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def train():

    random.seed(args.random_seed)
    np.random.seed(args.np_seed)
    torch.manual_seed(args.torch_seed)                 
    torch.cuda.manual_seed_all(args.cuda_seed) 
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, _, i_test, args4randpose = load_llff_data(args.datadir, args.factor,
                                                                      recenter=True, bd_factor=.75,
                                                                      spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]

        ### extra load for low resolution inference
        img4test, pose4test, _, render_poses, _, _ = load_llff_data(args.datadir, args.render_factor,
                                                                    recenter=True, bd_factor=.75,
                                                                    spherify=args.spherify)
        hwf4test = pose4test[0,:3,-1]
        pose4test = pose4test[:,:3,:4]

        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        img4test = images
        hwf4test = hwf
        pose4test = poses

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    hwf = get_hwf(hwf)
    hwf4test = get_hwf(hwf4test)
    H, W, focal = hwf

    if K is None:
        K = get_newK(hwf)

    if args.render_test:
        render_poses = np.array(pose4test[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = img4test[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{}_{:06d}'.format('res'+str(args.render_factor), 'test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, disps = render_path(200000, render_poses, hwf4test, args.chunk, args.N_gauss, render_kwargs_test, gt_imgs=images, savedir=testsavedir)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), to8b(disps / np.max(disps)), fps=30, quality=8)

            if args.render_test:

                psnrs = []
                ssims = []
                rgbs = torch.Tensor(rgbs)
                images = torch.Tensor(images)
                disps = torch.Tensor(disps / np.max(disps))
                
                for i in range(rgbs.shape[0]):
                    disp = disps[i]
                    rgb = rgbs[i]
                    gt = images[i]
                    
                    rgb8 = to8b(rgb.cpu().numpy())
                    filename = os.path.join(testsavedir, '{:03d}.png'.format(i))
                    imageio.imwrite(filename, rgb8)

                    disp8 = to8b(disp.cpu().numpy())
                    filename = os.path.join(testsavedir, '{:03d}_disp.png'.format(i))
                    imageio.imwrite(filename, disp8)

                psnrs += [mse2psnr(img2mse(rgb, gt))]
                ssims += [ssim(rgb, gt)]

            rec = rgbs.permute(0, 3, 1, 2)  # [N, 3, H, W]
            ref = images.permute(0, 3, 1, 2)  # [N, 3, H, W]
            rgb_rescale, gt_rescale = rescale(rec, -1, 1), rescale(ref, -1, 1)
            lpipses = lpips(rgb_rescale, gt_rescale).mean()

            psnrs = torch.stack(psnrs, dim=0).mean()
            ssims = torch.stack(ssims, dim=0).mean()

            print(f'PSNR: {str(psnrs.item())[:5]} | SSIM: {str(ssims.item())[:5]} | LPIPS: {str(lpipses.item())[:5]}')
            torch.save([psnrs, ssims, lpipses], os.path.join(testsavedir, f'psnr_{str(psnrs.item())[:5]}_ssim_{str(ssims.item())[:5]}_lpips_{str(lpipses.item())[:5]}.pth'))

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
        img4test = torch.Tensor(img4test).to(device)
    poses = torch.Tensor(poses).to(device)
    pose4test = torch.Tensor(pose4test).to(device)


    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    hem_pool = []
    num_pool = args.num_hem_pool
    if num_pool == 0:
        num_pool = 1
    iter_hem = int(N_rand / num_pool)
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        #############################################################################################
        # Sample random ray batch
        if (i-start) % args.xpose_iters == 0:
            pose = get_rand_pose(args4randpose)
        focal_ = focal# * (np.random.rand() + 1) if False else focal  # scale focal by [1, 2)

        if hem_pool and args.num_hem_pool > 0 and len(hem_pool) % iter_hem == 0:
            batch_rays = torch.cat(hem_pool, dim=1) # [2, 1024, 3]
            hem_pool = []
        else:
            rays_o, rays_d = get_rays_v2(H, W, focal_, pose[:3, :4])  # rays_o, rays_d shape: [H, W, 3]
        
            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            
            batch_rays = torch.stack([rays_o, rays_d], 0)
        #############################################################################################


        #####  Core optimization loop  #####
        rgb1, disp1, acc1, rgb2, disp2, acc2, con_loss, extras = render(i, H, W, K, chunk=args.chunk, 
                                                                        N_gauss=args.N_gauss, rays=batch_rays,
                                                                        verbose=i < 10, retraw=True, render_test=False,
                                                                        **render_kwargs_train)

        optimizer.zero_grad()
        img_loss1 = img2mse(rgb2, rgb1.detach())
        img_loss2 = img2mse(disp2, disp1.detach())
        img_loss3 = img2mse(acc1, acc2.detach())

        loss = img_loss1 + args.lambda_sigma * con_loss# + img_loss2 + img_loss3 + con_loss
        psnr = mse2psnr(img_loss1)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_gs'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            print('Test images shape for video', img4test.shape)
            downsample_factor = int(args.render_factor/args.factor)
            with torch.no_grad():
                rgbs, disps = render_path(i, render_poses, hwf4test, args.chunk, args.N_gauss, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            # torch.save(extras['raw1'].detach().cpu(), f'{basedir}/{expname}/raw1_iter_{i}.pth')
            # torch.save(extras['raw2'].detach().cpu(), f'{basedir}/{expname}/raw2_iter_{i}.pth')
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', pose4test[i_test].shape)
            
            with torch.no_grad():
                rgbs, disps = render_path(i, torch.Tensor(pose4test[i_test]).to(device), hwf4test, args.chunk, args.N_gauss, render_kwargs_test, gt_imgs=img4test[i_test], savedir=testsavedir)
            print('Saved test set')

            # filenames = [os.path.join(testsavedir, '{:03d}.png'.format(k)) for k in range(len(i_test))]

            # test_loss = img2mse(torch.Tensor(rgbs), img4test[i_test])
            # test_psnr = mse2psnr(test_loss)

            psnrs = []
            ssims = []
            rgbs = torch.Tensor(rgbs)
            gts = img4test[i_test]
            disps = torch.Tensor(disps / np.max(disps))
                
            for j in range(rgbs.shape[0]):
                disp = disps[j]
                rgb = rgbs[j]
                gt = gts[j]
                
                rgb8 = to8b(rgb.cpu().numpy())
                filename = os.path.join(testsavedir, '{:03d}.png'.format(j))
                imageio.imwrite(filename, rgb8)

                disp8 = to8b(disp.cpu().numpy())
                filename = os.path.join(testsavedir, '{:03d}_disp.png'.format(i))
                imageio.imwrite(filename, disp8)

                psnrs += [mse2psnr(img2mse(rgb, gt))]
                ssims += [ssim(rgb, gt)]

            rec = rgbs.permute(0, 3, 1, 2)  # [N, 3, H, W]
            ref = gts.permute(0, 3, 1, 2)  # [N, 3, H, W]
            rgb_rescale, gt_rescale = rescale(rec, -1, 1), rescale(ref, -1, 1)
            lpipses = lpips(rgb_rescale, gt_rescale).mean()

            psnrs = torch.stack(psnrs, dim=0).mean()
            ssims = torch.stack(ssims, dim=0).mean()

            print(f'PSNR: {str(psnrs.item())[:5]} | SSIM: {str(ssims.item())[:5]} | LPIPS: {str(lpipses.item())[:5]}')
            torch.save([psnrs, ssims, lpipses], f'{basedir}/{expname}/{i}_test_{str(psnrs.item())[:5]}_ssim_{str(ssims.item())[:5]}_lpips_{str(lpipses.item())[:5]}.pth')

            tqdm.write(f"[TEST] Iter: {i} Loss: {loss.item()}  Test PSNR: {psnrs.item()}")

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss1: {str(img_loss1.item())[:6]} | Constraint: {str(con_loss.item())[:6]} | PSNR1: {str(psnr.item())[:7]}")

        if args.num_hem_pool > 0:
            # print(batch_rays.shape, rgb2.shape) # [2, 1024, 3], [1024, 3]
            _, indices = torch.sort(torch.mean((rgb2.detach() - rgb1.detach())**2, dim=1))
            hard_indices = indices[-num_pool:]
            hard_rays = batch_rays[:, hard_indices, :]
            hem_pool += [hard_rays]
            # print(hard_rays.shape) # [2, 16, 3]

        global_step += 1


if __name__=='__main__':

    train()
