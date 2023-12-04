import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Two heads for seperate sigma and rgb input and output
class Branch2NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """ 
        """
        super(Branch2NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips

        self.input_linear1 = nn.Sequential(
            nn.Linear(1008, W//2),
            nn.ReLU()
            )
        
        self.input_linear2 = nn.Sequential(
            nn.Linear(input_ch-1008, W//2),
            nn.ReLU()
            )

        self.pts_linears = nn.ModuleList(
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-2)])

        self.notres_layers = [i for i in range(D-2) if i in self.skips]
        self.concat_layers = [i-1 for i in range(D-2) if i in self.skips]

        self.output_linear1 = nn.Sequential(
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, output_ch//4)
            )
        
        self.output_linear2 = nn.Sequential(
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, output_ch//4*3)
            )

    def forward(self, x):
        input_pt = x[..., :1008]
        input_od = x[..., 1008:]

        x_pt = self.input_linear1(input_pt)
        x_od = self.input_linear2(input_od)

        h = torch.cat([x_pt, x_od], dim=-1)
        for i, l in enumerate(self.pts_linears):
            if i not in self.notres_layers:
                h = self.pts_linears[i](h) + h
            else:
                h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.concat_layers:
                h = torch.cat([x, h], -1)

        outputs1 = self.output_linear1(h)
        outputs2 = self.output_linear2(h)
        outputs = torch.cat([outputs1, outputs2], dim=-1)

        return outputs

# Two heads for seperate sigma and rgb output
class Head2NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """ 
        """
        super(Head2NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-2)])

        self.cat_layers = [i+1 for i in range(D-1) if i in self.skips] + [0]

        # self.output_linear1 = nn.Linear(W, output_ch//4)
        # self.output_linear2 = nn.Linear(W, output_ch//4*3)

        self.output_linear1 = nn.Sequential(
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, output_ch//4)
            )
        
        self.output_linear2 = nn.Sequential(
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, output_ch//4*3)
            )

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            if i not in self.cat_layers:
                h = self.pts_linears[i](h) + h
            else:
                h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        outputs1 = self.output_linear1(h)
        outputs2 = self.output_linear2(h)
        outputs = torch.cat([outputs1, outputs2], dim=-1)

        return outputs

# NeRF with residual connection and input concatenation in middle layers
class NewNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """ 
        """
        super(NewNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.cat_layers = [i+1 for i in range(D-1) if i in self.skips] + [0]

        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            if i not in self.cat_layers:
                h = self.pts_linears[i](h) + h
            else:
                h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        outputs = self.output_linear(h)

        return outputs

# Multi-resolution NeRF
class MResNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """ 
        """
        super(MResNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.res_layers = [i+1 for i in range(D-1) if i in self.skips] + [0]

        self.output_linear1 = nn.Linear(W, output_ch)
        self.output_linear2 = nn.Linear(W, output_ch)

    def forward(self, x, depth4):
        h = x
        for i, l in enumerate(self.pts_linears):
            if i not in self.res_layers:
                h = self.pts_linears[i](h) + h
            else:
                h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
            if depth4:
                if i == 3:
                    h_depth4 = h
        if depth4:
            outputs = self.output_linear1(h_depth4)
        else:
            outputs = self.output_linear2(h)

        return outputs

# NeRF with only residual connections (old version)
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
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

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

# NeRF-style NLP to output RGB
class RGBNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, num_Gaussian=8, skips=[4]):
        """ 
        """
        super(RGBNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.num_Gau = num_Gaussian
        self.skips = skips
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.output_linear = nn.Linear(W, 3)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        outputs = self.output_linear(h)

        return outputs

# Official NeRF
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

##############################################################################################################
# NeRF_v3 copied from R2L
class ResMLP(nn.Module):

    def __init__(self,
                 width,
                 inact=nn.ReLU(True),
                 outact=None,
                 res_scale=1,
                 n_learnable=2):
        '''inact is the activation func within block. outact is the activation func right before output'''
        super(ResMLP, self).__init__()
        m = [nn.Linear(width, width)]
        for _ in range(n_learnable - 1):
            if inact is not None: m += [inact]
            m += [nn.Linear(width, width)]
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.outact = outact

    def forward(self, x):
        x = self.body(x).mul(self.res_scale) + x
        if self.outact is not None:
            x = self.outact(x)
        return x


def get_activation(act):
    if act.lower() == 'relu':
        func = nn.ReLU(inplace=True)
    elif act.lower() == 'lrelu':
        func = nn.LeakyReLU(inplace=True)
    elif act.lower() == 'none':
        func = None
    else:
        raise NotImplementedError
    return func

# NeRF_v3 copied from R2L
class NeRF_v3_2(nn.Module):
    '''Based on NeRF_v3, move positional embedding out'''

    def __init__(self, args, input_dim, output_dim):
        super(NeRF_v3_2, self).__init__()
        self.args = args
        D, W = args.netdepth_gs, args.netwidth_gs

        # get network width
        if args.layerwise_netwidths:
            Ws = [int(x) for x in args.layerwise_netwidths.split(',')] + [3]
            print('Layer-wise widths are given. Overwrite args.netwidth')
        else:
            Ws = [W] * (D - 1) + [3]

        # the main non-linear activation func
        act = get_activation(args.act)

        # head
        self.input_dim = input_dim
        self.head = nn.Sequential(*[nn.Linear(input_dim, Ws[0]), act])

        # body
        body = []
        for i in range(1, D - 1):
            body += [nn.Linear(Ws[i - 1], Ws[i]), act]

        # >>> new implementation of the body. Will replace the above
        if hasattr(args, 'trial'):
            inact = get_activation(args.trial.inact)
            outact = get_activation(args.trial.outact)
            if args.trial.body_arch in ['resmlp']:
                n_block = (
                    D - 2
                ) // 2  # 2 layers in a ResMLP, deprecated since there can be >2 layers in a block, use --trial.n_block
                if args.trial.n_block > 0:
                    n_block = args.trial.n_block
                body = [
                    ResMLP(W,
                           inact=inact,
                           outact=outact,
                           res_scale=args.trial.res_scale,
                           n_learnable=args.trial.n_learnable)
                    for _ in range(n_block)
                ]
            elif args.trial.body_arch in ['mlp']:
                body = []
                for i in range(1, D - 1):
                    body += [nn.Linear(Ws[i - 1], Ws[i]), act]
        # <<<

        self.body = nn.Sequential(*body)

        # tail
        self.tail = nn.Linear(
            input_dim, output_dim) if args.linear_tail else nn.Sequential(
                *[nn.Linear(Ws[D - 2], output_dim)])

    def forward(self, x):  # x: embedded position coordinates
        if x.shape[-1] != self.input_dim:  # [N, C, H, W]
            x = x.permute(0, 2, 3, 1)
        x = self.head(x)
        x = self.body(x) + x if self.args.use_residual else self.body(x)
        return self.tail(x)


model_gs_dict = {'res_cat':   NewNeRF, 
                 '2head':     Head2NeRF, 
                 'res':       MYNeRF, 
                 'r2l':       NeRF_v3_2, 
                 '2branch':   Branch2NeRF, 
                 'multi-res': MResNeRF, 
                 'rgb':       RGBNeRF}