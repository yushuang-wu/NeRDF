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
from input_emb import *

from load_llff import *
from load_blender import load_blender_data

from option import config_parser
from functools import partial

import lpips as lpips_
from utils.ssim_torch import ssim as ssim_

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
    if not os.path.exists(args.basedir):
        os.makedirs(args.basedir)
else:
    print('Local Training!')


def batchify_rays(epoch, rays_flat, hwf, N_gauss, depth4, render_test, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(epoch, rays_flat[i:i+chunk], hwf, N_gauss, depth4, render_test, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret_new = {k : torch.cat(all_ret[k], 0) for k in all_ret if k != 'constrain'}
    all_ret_new['constrain'] = 0
    for i in all_ret['constrain']:
        all_ret_new['constrain'] += i
    return all_ret_new


def render(epoch, H, W, K, chunk=1024*32, depth4=False, 
           N_gauss=4, rays=None, c2w=None, ndc=True,
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
    all_ret = batchify_rays(epoch, rays, hwf, N_gauss, depth4, render_test, chunk, **kwargs)
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

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        # print(i, time.time() - t)
        t = time.time()
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
    model1 = NeRF(D=args.netdepth, W=args.netwidth,
                  input_ch=input_ch, output_ch=output_ch, skips=skips,
                  input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    model2 = NeRF(D=args.netdepth, W=args.netwidth,
                  input_ch=input_ch, output_ch=output_ch, skips=skips,
                  input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    # grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine1 = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                           input_ch=input_ch, output_ch=output_ch, skips=skips,
                           input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        model_fine2 = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        # grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network1(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    ##################################################################################################
    gauss = MResNeRF(D=args.netdepth_gs, W=args.netwidth_gs,
                     input_ch=args.input_ch, output_ch=args.N_gauss*3*4, skips=skips).to(device)
    # gauss = NeRF_v3_2(args, input_dim=args.input_ch, output_dim=args.N_gauss*3*4).to(device)
    grad_vars = list(gauss.parameters())

    network_query_gs = lambda inputs, viewdirs, network_fn, depth4 : run_network2d4(inputs, viewdirs, network_fn, depth4,
                                                                 embed_fn=embed_fn_gs,
                                                                 embeddirs_fn=embeddirs_fn_gs,
                                                                 netchunk=args.netchunk)
    # network_query_gs = lambda pts, rays_o, viewdirs, network_fn : run_network2ptsod(pts, rays_o, viewdirs, network_fn,
    #                                                              embed_fn=embed_fn_gs,
    #                                                              embeddirs_fn=embeddirs_fn_gs,
    #                                                              netchunk=args.netchunk)
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
    teacher_ckpt2 = 'models/fern_down16.tar'
    print('Teacher ckpt', args.teacher_ckpt, teacher_ckpt2)
    if not args.render_test:
        ckpt1 = torch.load(args.teacher_ckpt)
        model1.load_state_dict(ckpt1['network_fn_state_dict'])
        model_fine1.load_state_dict(ckpt1['network_fine_state_dict'])

        ckpt2 = torch.load(teacher_ckpt2)
        model2.load_state_dict(ckpt2['network_fn_state_dict'])
        model_fine2.load_state_dict(ckpt2['network_fine_state_dict'])

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        gauss.load_state_dict(ckpt['network_fn_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'network_query_gs' : network_query_gs,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine1' : model_fine1,
        'network_fine2' : model_fine2,
        'N_samples' : args.N_samples,
        'network_fn1' : model1,
        'network_fn2' : model2,
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
                depth4,
                render_test,
                network_fn1,
                network_fn2,
                network_fine1,
                network_fine2,
                network_gs,
                network_query_fn,
                network_query_gs,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
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
    network_fn = network_fn2 if depth4 else network_fn1
    network_fine = network_fine2 if depth4 else network_fine1
    
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

    raw_org2 = network_query_gs(pts2, viewdirs, network_gs, depth4)  # [N_rays, 3]
    # raw_org2 = network_query_gs(pts2, rays_o, viewdirs, network_gs)  # [N_rays, 3]
    rgb_map2, disp_map2, acc_map2, *_ = MYraw2outputs(epoch, raw_org2, z_vals_pred, near, far, rays_d, N_gauss, raw_noise_std, white_bkgd, pytest=pytest)
    
    if render_test:
        constrain = torch.tensor(0)

    elif args.sigma_constrain or args.dev_regularization:
        constrain1, constrain2 = torch.tensor(0), torch.tensor(0)

        if args.sigma_constrain:
            sigma2, rgb2 = get_sigma(epoch, raw_org2, z_vals, near, far, rays_d, N_gauss, raw_noise_std, white_bkgd, pytest=pytest)

            ### sigma constrain ###
            sigma1_norm = F.normalize(torch.clip(sigma1, min=0), dim=-1)
            sigma2_norm = F.normalize(torch.clip(sigma2, min=0), dim=-1)
            sigma_const = F.mse_loss(sigma2_norm, sigma1_norm.detach())
            constrain1 = sigma_const * 1.0
            
            ### rgb constrain ###
            # rgb_const = F.mse_loss(rgb2, rgb1.detach())
            # constrain = (sigma_const + rgb_const) * 1.0

            ### sigma peak position ###
            # peak1 = sigma1.max(1)[1].float().detach()
            # peak2 = sigma2.max(1)[1].float()
            # constrain = F.l1_loss(peak2, peak1)

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
    ### resolution1
    images1, poses1, bds1, _, _, args4randpose1 = load_llff_data(args.datadir, args.factor,
                                                                 recenter=True, bd_factor=.75,
                                                                 spherify=args.spherify)
    hwf1 = poses1[0,:3,-1]
    poses1 = poses1[:,:3,:4]

    ### resolution2
    images2, poses2, bds2, _, _, args4randpose2 = load_llff_data(args.datadir, 16,
                                                                 recenter=True, bd_factor=.75,
                                                                 spherify=args.spherify)
    hwf2 = poses2[0,:3,-1]
    poses2 = poses2[:,:3,:4]

    ### extra load for low resolution inference
    img4test, pose4test, _, render_poses, i_test, _ = load_llff_data(args.datadir, args.render_factor,
                                                                     recenter=True, bd_factor=.75,
                                                                     spherify=args.spherify)
    hwf4test = pose4test[0,:3,-1]
    pose4test = pose4test[:,:3,:4]

    print('Loaded llff', images1.shape, render_poses.shape, hwf1, args.datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print('Auto LLFF holdout,', args.llffhold)
        i_test = np.arange(img4test.shape[0])[::args.llffhold]

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(img4test.shape[0])) if
                       (i not in i_test and i not in i_val)])

    print('DEFINING BOUNDS')
    if args.no_ndc:
        near = np.ndarray.min(bds1) * .9
        far = np.ndarray.max(bds1) * 1.
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)

    # Cast intrinsics to right types
    hwf1 = get_hwf(hwf1)
    hwf2 = get_hwf(hwf2)
    
    H1, W1, focal1 = hwf1
    H2, W2, focal2 = hwf2

    K1 = get_newK(hwf1)
    K2 = get_newK(hwf2)

    hwf4test = get_hwf(hwf4test)

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

            rgbs, _ = render_path(200000, render_poses, hwf4test, args.chunk, args.N_gauss, render_kwargs_test, gt_imgs=images, savedir=testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            psnrs = []
            ssims = []
            rgbs = torch.Tensor(rgbs)
            images = torch.Tensor(images)
            
            for i in range(rgbs.shape[0]):
                rgb = rgbs[i]
                gt = images[i]
                
                rgb8 = to8b(rgb.cpu().numpy())
                filename = os.path.join(testsavedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)

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
        images1 = torch.Tensor(images1).to(device)
        images2 = torch.Tensor(images2).to(device)
        img4test = torch.Tensor(img4test).to(device)

    poses1 = torch.Tensor(poses1).to(device)
    poses2 = torch.Tensor(poses2).to(device)
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
        if (i-start) % args.xpose_iters == 0:
            pose1 = get_rand_pose(args4randpose1)
            pose2 = get_rand_pose(args4randpose2)
        # focal_ = focal * (np.random.rand() + 1) if False else focal  # scale focal by [1, 2)

        if hem_pool and args.num_hem_pool > 0 and len(hem_pool) % iter_hem == 0:
            batch_rays = torch.cat(hem_pool, dim=1) # [2, 1024, 3]
            hem_pool = []
        else:
            batch_rays1 = get_batchrays(args, pose1, hwf1, N_rand, i, start)
            batch_rays2 = get_batchrays(args, pose2, hwf2, N_rand, i, start)
        #############################################################################################


        #####  Core optimization loop  #####
        rgb11, disp11, acc11, rgb21, disp21, acc21, con_loss1, extras1 = render(
            i, H1, W1, K1, 
            chunk=args.chunk, depth4=False,
            N_gauss=args.N_gauss, rays=batch_rays1,
            verbose=i < 10, retraw=True, render_test=False, 
            **render_kwargs_train
            )

        rgb12, disp12, acc12, rgb22, disp22, acc22, con_loss2, extras2 = render(
            i, H2, W2, K2, 
            chunk=args.chunk, depth4=True,
            N_gauss=args.N_gauss, rays=batch_rays2,
            verbose=i < 10, retraw=True, render_test=False, 
            **render_kwargs_train
            )

        optimizer.zero_grad()
        img_loss1 = img2mse(rgb21, rgb11.detach())
        img_loss2 = img2mse(rgb22, rgb12.detach())

        loss = img_loss1 + img_loss2 + con_loss1 + con_loss2
        psnr1 = mse2psnr(img_loss1)
        psnr2 = mse2psnr(img_loss2)

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
            downsample_factor = int(args.render_factor/args.factor)
            with torch.no_grad():
                rgbs, disps = render_path(i, torch.Tensor(pose4test[i_test]).to(device), hwf4test, args.chunk, args.N_gauss, render_kwargs_test, gt_imgs=img4test[i_test], savedir=testsavedir)
            print('Saved test set')

            filenames = [os.path.join(testsavedir, '{:03d}.png'.format(k)) for k in range(len(i_test))]

            test_loss = img2mse(torch.Tensor(rgbs), img4test[i_test])
            test_psnr = mse2psnr(test_loss)

            tqdm.write(f"[TEST] Iter: {i} Loss: {loss.item()}  Test PSNR: {test_psnr.item()}")
            # torch.save(test_psnr, f'{basedir}/{expname}/{i}_{str(psnr2.item())[:5]}_{str(test_psnr.item())[:5]}.psnr')
            torch.save(test_psnr, f'{basedir}/{expname}/{i}_test_{str(test_psnr.item())[:5]}.psnr')

        if i%args.i_print==0:
            # tqdm.write(f"[TRAIN] Iter: {i} Loss1: {str(img_loss1.item())[:6]} | PSNR1: {str(psnr1.item())[:7]} | Loss2: {str(img_loss2.item())[:6]} | PSNR2: {str(psnr2.item())[:7]} | Loss3: {str(img_loss3.item())[:6]}")
            tqdm.write(f"[TRAIN] Iter: {i} Loss1: {str(img_loss1.item())[:6]} | Loss2: {str(img_loss2.item())[:6]} | PSNR1: {str(psnr1.item())[:7]} | PSNR2: {str(psnr2.item())[:7]}")

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
