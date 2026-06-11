##################################################
# This implementation follows the methodology described in:
# Lee et al. "AZ-NAS: Assembling Zero-Cost Proxies for Network Architecture Search", CVPR 2024.
##################################################

import os
import torch
from sympy import false

import os
import torch
import datetime
import sys

log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class DualLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.console = sys.stdout
        self.log_file = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.console.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

original_stdout = sys.stdout


import os, sys, time, glob, random, argparse
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import torch.nn as nn
import time
import tqdm
import scipy.stats as stats  
import matplotlib.pyplot as plt
import pickle 

# XAutoDL
from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, get_nas_search_loaders 
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy  
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time  
from xautodl.models import get_search_spaces

# API
from nats_bench import create  
# custom modules
from custom.tss_model import TinyNetwork  
from xautodl.models.cell_searchs.genotypes import Structure
from ZeroShotProxy import *

parser = argparse.ArgumentParser("Training-free NAS on NAS-Bench-201 (NATS-Bench-TSS)")
parser.add_argument("--data_path", type=str, default='./cifar.python', help="The path to dataset")
parser.add_argument("--dataset", type=str, default='cifar100', choices=["cifar10", "cifar100", "ImageNet16-120"],
                    help="Choose between Cifar10/100 and ImageNet-16.")

# channels and number-of-cells
parser.add_argument("--search_space", type=str, default='tss', help="The search space name.")
parser.add_argument("--config_path", type=str, default='./configs/nas-benchmark/algos/weight-sharing.config',
                    help="The path to the configuration.")
parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
parser.add_argument("--num_cells", type=int, default=5, help="The number of cells in one stage.")
parser.add_argument("--affine", type=int, default=1, choices=[0, 1],
                    help="Whether use affine=True or False in the BN layer.")
parser.add_argument("--track_running_stats", type=int, default=0, choices=[0, 1],
                    help="Whether use track_running_stats or not in the BN layer.")

# log
parser.add_argument("--print_freq", type=int, default=200, help="print frequency (default: 200)")

# custom
parser.add_argument("--gpu", type=int, default=0, help="")
parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
parser.add_argument("--api_data_path", type=str, default="./api_data/NATS-tss-v1_0-3ffb9-simple/", help="")
parser.add_argument("--save_dir", type=str, default='./results/tmp', help="Folder to save checkpoints and log.")
parser.add_argument('--zero_cost_score', type=str, default='singular',
                    choices=['az_nas', 'zico', 'zen', 'gradnorm', 'naswot', 'synflow', 'snip',
                             'grasp', 'te_nas', 'entropic', 'singular'])
parser.add_argument("--rand_seed", type=int, default=1, help="manual seed (we use 1-to-5)")
parser.add_argument("--log_dir", type=str, default='./logs', help="Directory to save logs and images")
args = parser.parse_args()

if args.rand_seed is None or args.rand_seed < 0:
    args.rand_seed = random.randint(1, 100000)

log_filename = f"{args.zero_cost_score}_{args.dataset}_{args.rand_seed}_{timestamp}.txt"
log_filepath = os.path.join(args.log_dir, log_filename)

dual_logger = DualLogger(log_filepath)
sys.stdout = dual_logger

print(f"seed: {args.rand_seed}")
print(f"args: {args}")
xargs = args

assert torch.cuda.is_available(), "CUDA is not available."
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(xargs.workers)
prepare_seed(xargs.rand_seed)
logger = prepare_logger(args)

## API api_data_path
api = create(xargs.api_data_path, xargs.search_space, fast_mode=True, verbose=False)
logger.log("Create API = {:} done".format(api))

## data  xshape=[batch,channel,height,width]
train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
config = load_config(xargs.config_path, {"class_num": class_num, "xshape": xshape}, logger)
search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data,
                                                                   valid_data,
                                                                   xargs.dataset,
                                                                   "./configs/nas-benchmark/",
                                                                   (config.batch_size, config.test_batch_size),
                                                                   xargs.workers, )
logger.log("||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(xargs.dataset,
                                                                                                       len(search_loader),
                                                                                                       len(valid_loader),
                                                                                               config.batch_size))
logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

## model
search_space = get_search_spaces(xargs.search_space, "nats-bench")
logger.log("search space : {:}".format(search_space))

device = torch.device('cuda:{}'.format(xargs.gpu))

def random_genotype(max_nodes, op_names):

    genotypes = []
    for i in range(1, max_nodes):
        xlist = []
        for j in range(i):
            node_str = "{:}<-{:}".format(i, j)
            op_name = random.choice(op_names)
            xlist.append((op_name, j))
        genotypes.append(tuple(xlist))
    arch = Structure(genotypes)
    return arch


real_input_metrics = ['zico', 'snip', 'grasp', 'te_nas']


def search_find_best(xargs, xloader, n_samples=None, archs=None):
    logger.log("Searching with {}".format(xargs.zero_cost_score.lower()))
    score_fn_name = "compute_{}_score".format(xargs.zero_cost_score.lower())
    score_fn = globals().get(score_fn_name)

    input_, target_ = next(iter(xloader))
    resolution = input_.size(2)
    batch_size = input_.size(0)
    zero_cost_score_dict = None
    arch_list = []

    if xargs.zero_cost_score.lower() in real_input_metrics:
        trainloader = train_loader
        print("real input")
    else:
        trainloader = None
        print("noise input")

   
    if archs is None and n_samples is not None:
        all_time = []
        all_mem = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for i in tqdm.tqdm(range(n_samples)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            arch = random_genotype(xargs.max_nodes, search_space)
            network = TinyNetwork(xargs.channel, xargs.num_cells, arch, class_num)
            network = network.to(device)
            network.train()

            start.record()
            info_dict = score_fn.compute_nas_score(network, gpu=xargs.gpu, trainloader=trainloader,
                                                   resolution=resolution, batch_size=batch_size, verbose=False,
                                                   dataset=xargs.dataset)
            end.record()
            torch.cuda.synchronize()
            all_time.append(start.elapsed_time(end))
            all_mem.append(torch.cuda.max_memory_allocated() / 1e9)

            arch_list.append(arch)
            if zero_cost_score_dict is None:
                zero_cost_score_dict = dict()
                for k in info_dict.keys():
                    zero_cost_score_dict[k] = []
            for k, v in info_dict.items():
                zero_cost_score_dict[k].append(v)

        avg_time = np.mean(all_time)
        avg_mem = np.mean(all_mem)
        max_mem = np.max(all_mem)


        logger.log("------Runtime------")
        logger.log(f"All: {avg_time:.5f} ms")
        logger.log("------Avg Mem------")
        logger.log(f"All: {avg_mem:.5f} GB")
        logger.log("------Max Mem------")
        logger.log(f"All: {max_mem:.5f} GB")

    return arch_list, zero_cost_score_dict, avg_time if 'avg_time' in locals() else 0


archs, results, search_time = search_find_best(xargs, train_loader, n_samples=1000)


def get_results_from_api(api, arch, dataset='cifar10'):
    dataset_candidates = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
    assert dataset in dataset_candidates

    index = api.query_index_by_arch(arch)
    api._prepare_info(index)
    archresult = api.arch2infos_dict[index]['200']

    if dataset == 'cifar10-valid':
        acc = archresult.get_metrics(dataset, 'x-valid', iepoch=None, is_random=False)['accuracy']
    elif dataset == 'cifar10':
        acc = archresult.get_metrics(dataset, 'ori-test', iepoch=None, is_random=False)['accuracy']
    else:
        acc = archresult.get_metrics(dataset, 'x-test', iepoch=None, is_random=False)['accuracy']

    flops = archresult.get_compute_costs(dataset)['flops']
    params = archresult.get_compute_costs(dataset)['params']

    return acc, flops, params


api_valid_accs, api_flops, api_params = [], [], []

for a in archs:
    valid_acc, flops, params = get_results_from_api(api, a, xargs.dataset)
    api_valid_accs.append(valid_acc)
    api_flops.append(flops)
    api_params.append(params)

# ==================== ====================

print("\n" + "=" * 80)
print("details")
print("=" * 80)

# print("\n=== space ===")
# print(f" {xargs.dataset}")
# print(f"{xargs.data_path}")
# print(f"{xargs.search_space}")
# print(f"{xargs.max_nodes}")
# print(f"{xargs.channel}")
# print(f"{xargs.num_cells}")
# print(f"{xargs.rand_seed}")
# print(f"{xargs.zero_cost_score}")
# print(f"GPU: {xargs.gpu}")
#
# print(f"\n=== search ===")
# print(f" {search_time:.5f} ms")
# print(f"{len(archs)}")
# print(f"{search_time * len(archs) / 1000:.2f} s")

print("\n=== top ===")
max_acc_idx = np.argmax(api_valid_accs)
max_acc = api_valid_accs[max_acc_idx]
max_acc_arch = archs[max_acc_idx]

print(f"best acc:{max_acc:.4f}%")
print(f"best idx:{max_acc_idx}")
print(f"best arch:{max_acc_arch}")
print(f"flops{api_flops[max_acc_idx]:.4f}")
print(f"params{api_params[max_acc_idx]:.4f}")

if api is not None:
    arch_info = api.query_by_arch(max_acc_arch, "200")
    print(f"info:{arch_info}")

print("\n" + "=" * 80)
print("zero-cost metric")
print("=" * 80)

fig_scale = 1.1

if xargs.zero_cost_score.lower() == 'te_nas':
    print(f"\n===  {xargs.zero_cost_score} ===")
    rank_agg = None
    for k in results.keys():
        print(f" {k}")
        if rank_agg is None:
            rank_agg = stats.rankdata(results[k])
        else:
            rank_agg = rank_agg + stats.rankdata(results[k])

    best_idx = np.argmax(rank_agg)
    best_arch, acc = archs[best_idx], api_valid_accs[best_idx]

    print(f"arch:{best_arch}")
    print(f"acc:{acc:.4f}%")
    print(f"idx:{best_idx}")

    if api is not None:
        arch_info = api.query_by_arch(best_arch, "200")
        print(f"info:{arch_info}")

    x = stats.rankdata(rank_agg)
    y = stats.rankdata(api_valid_accs)
    kendalltau = stats.kendalltau(x, y)
    spearmanr = stats.spearmanr(x, y)
    pearsonr = stats.pearsonr(x, y)

    print(f"Kendall's Tau: {kendalltau[0]:.6f} (p-value: {kendalltau[1]:.6f})")
    print(f"Pearson R: {pearsonr[0]:.6f} (p-value: {pearsonr[1]:.6f})")
    print(f"Spearman R: {spearmanr[0]:.6f} (p-value: {spearmanr[1]:.6f})")


else:
    for k, v in results.items():
        print(f"\n===  {k}  ===")
        best_idx = np.argmax(v)
        best_arch, acc = archs[best_idx], api_valid_accs[best_idx]

        print(f"{k} ")
        print(f"{best_arch}")
        print(f"{acc:.4f}%")
        print(f"{best_idx}")

        if api is not None:
            arch_info = api.query_by_arch(best_arch, "200")
            print(f"{arch_info}")

        x = stats.rankdata(v)
        y = stats.rankdata(api_valid_accs)
        kendalltau = stats.kendalltau(x, y)
        spearmanr = stats.spearmanr(x, y)
        pearsonr = stats.pearsonr(x, y)

        print(f"Kendall's Tau: {kendalltau[0]:.6f} (p-value: {kendalltau[1]:.6f})")
        print(f"Pearson R: {pearsonr[0]:.6f} (p-value: {pearsonr[1]:.6f})")
        print(f"Spearman R: {spearmanr[0]:.6f} (p-value: {spearmanr[1]:.6f})")


dual_logger.close()
sys.stdout = original_stdout
