import torch
path = '../logs/fern_gauss_norm_mse/020000.tar'
a = torch.load(path)

b = a['network_fn_state_dict']
torch.save(b, 'example_ckpt.tar')