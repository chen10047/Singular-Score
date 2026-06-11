"""
Singular Score adapted for NAS‑Bench‑201
"""
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np


def kaiming_normal_fanin_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def kaiming_uniform_fanin_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def kaiming_uniform_fanout_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def xavier_normal_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def xavier_uniform_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def plain_normal_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def plain_uniform_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-0.1, b=0.1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    elif method == 'kaiming_uni_fanin':
        model.apply(kaiming_uniform_fanin_init)
    elif method == 'kaiming_uni_fanout':
        model.apply(kaiming_uniform_fanout_init)
    elif method == 'xavier_norm':
        model.apply(xavier_normal_init)
    elif method == 'xavier_uni':
        model.apply(xavier_uniform_init)
    elif method == 'plain_norm':
        model.apply(plain_normal_init)
    elif method == 'plain_uni':
        model.apply(plain_uniform_init)
    else:
        raise NotImplementedError
    return model

def compute_real_activation_maps(model, input_tensor):
    activation_maps = []
    hooks = []

    def activation_hook(module, input, output):
        activation_maps.append(output.detach().clone())

    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU)):
            hooks.append(module.register_forward_hook(activation_hook))

    with torch.no_grad():
        _ = model(input_tensor)

    for hook in hooks:
        hook.remove()

    return activation_maps

def compute_nas_score(model, gpu, trainloader, resolution, batch_size, init_method='kaiming_norm_fanin', fp16=False,
                      verbose=False, dataset=None):
    model.train()
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
        model = model.to(device)
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    init_model(model, init_method)

    if trainloader is None:
        input_ = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
    else:
        input_ = next(iter(trainloader))[0].to(device)

    ################################
    activation_maps = compute_real_activation_maps(model, input_)

    singular_scores = []
    for i, activation in enumerate(activation_maps):
        b, c, h, w = activation.size()

        activation_matrix = activation.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)

        center_data = True
        if center_data:
            mean = activation_matrix.mean(dim=0, keepdim=True)
            activation_matrix = activation_matrix - mean

        U, S, V = torch.svd(activation_matrix)
        singular_values = S.cpu().numpy()

        prob_s = singular_values / (singular_values.sum() + 1e-8)

        entropy = -np.sum(prob_s * np.log(prob_s + 1e-8))

        singular_scores.append(entropy)

    singular_score = np.sum(singular_scores)
    # print(singular_score)
    info = {}
    info['singular'] = float(singular_score) if not np.isnan(singular_score) else -np.inf

    return info