##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08 ##########################
##############################################################################
# NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size #
##############################################################################
"""The official Application Programming Interface (API) for NATS-Bench."""


from typing import Text, Optional # 导入提示类型理解
from nats_bench.api_size import NATSsize # 导入尺寸搜索空间 API
from nats_bench.api_topology import NATStopology # 导入拓扑搜索空间 API
from nats_bench.api_utils import ArchResults # 导入架构评估结果 API
from nats_bench.api_utils import pickle_load # 导入序列化加载函数
from nats_bench.api_utils import pickle_save # 导入序列化保存函数
from nats_bench.api_utils import ResultsCount # 导入评估结果统计类

# API的历史版本
NATS_BENCH_API_VERSIONs = [
    "v1.0",  # [2020.08.31] initialize
    "v1.1",  # [2020.12.20] add unit tests
    "v1.2",  # [2021.03.17] black re-formulate
    "v1.3",  # [2021.04.08] fix find_best issue for fast_mode=True
    "v1.4",  # [2021.04.30] add topology_str2structure
    "v1.5",  # [2021.12.09] make simulate_train_eval more robust
    "v1.6",  # [2022.01.19] fix the inconsistent flop/params which is caused by a legacy (weight migration) issue
    "v1.7",  # [2022.03.25] relax enforce_all kwargs and add a test
    "v1.8",  # [2022.10.06] fix bugs at issues/44
]

NATS_BENCH_SSS_NAMEs = ("sss", "size")
NATS_BENCH_TSS_NAMEs = ("tss", "topology")

def version():
    return NATS_BENCH_API_VERSIONs[-1]


def create(file_path_or_dict, search_space, fast_mode=False, verbose=True):

    if search_space in NATS_BENCH_TSS_NAMEs:
        return NATStopology(file_path_or_dict, fast_mode, verbose)
    elif search_space in NATS_BENCH_SSS_NAMEs:
        return NATSsize(file_path_or_dict, fast_mode, verbose)
    else:
        raise ValueError("invalid search space : {:}".format(search_space))


def search_space_info(main_tag: Text, aux_tag: Optional[Text]):

    nats_sss = dict(candidates=[8, 16, 24, 32, 40, 48, 56, 64], num_layers=5)

    nats_tss = dict(
        op_names=[
            "none",
            "skip_connect",
            "nor_conv_1x1",
            "nor_conv_3x3",
            "avg_pool_3x3",
        ],
        num_nodes=4,
    )

    if main_tag == "nats-bench":
        if aux_tag in NATS_BENCH_SSS_NAMEs:
            return nats_sss
        elif aux_tag in NATS_BENCH_TSS_NAMEs:
            return nats_tss
        else:
            raise ValueError("Unknown auxiliary tag: {:}".format(aux_tag))
    elif main_tag == "nas-bench-201":
        if aux_tag is not None:
            raise ValueError("For NAS-Bench-201, the auxiliary tag should be None.")
        return nats_tss
    else:
        raise ValueError("Unknown main tag: {:}".format(main_tag))
