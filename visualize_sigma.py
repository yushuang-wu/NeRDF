import torch
import matplotlib.pyplot as plt
import random

root = 'logs/fern_gauss_norm_mse/sigma_vis'
sigma1 = torch.load(f'{root}/sigma1.pth')
sigma2 = torch.load(f'{root}/sigma2.pth')
z_vals = torch.load(f'{root}/z_vals.pth')
sigma1[sigma1<0] = 0
sigma2[sigma2<0] = 0

print(sigma1.shape, sigma2.shape, z_vals.shape)


# index = random.choice(list(range(2048)))
# print(index)
index = 944
x = z_vals[index].numpy().tolist()
assert x == sorted(x)
# print(x)
# x_d = {x[i]: i for i in range(len(x))}
# x_d = dict(sorted(x_d.items()))
# print(x_d.keys())

y_1 = sigma1[index].numpy().tolist()
y_2 = sigma2[index].numpy().tolist()

plt.figure()
plt.plot(x, y_1, color='r', label='Teacher NeRF')
plt.plot(x, y_2, color='b', label='NeRDF')

plt.legend(loc = 0, prop = {'size':16})
plt.yticks(size = 14)
plt.xticks(size = 14)
# plt.xlabel('t', size = 14)
# plt.ylabel('y_label', size = 14)
plt.savefig(f'{root}/sigma_compare.png', dpi=100)
