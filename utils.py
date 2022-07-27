import numpy as onp
import os
from scipy.stats import bernoulli
import random
from EliasOmega import encode


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    onp.random.seed(seed)


def rotate_func(l, n):
    return l[n:] + l[:n]


def ind_sparsification(x, p):
    x_sp = onp.zeros_like(x)
    idx = []
    for j in range(len(x)):
        if bernoulli.rvs(p[j]):
            idx.append(j)
    x_sp[idx] = x[idx]
    return x_sp


def standard_dithering(x, s, p):
    # s is scalar: n_levels
    y = onp.abs(x) / (onp.linalg.norm(x, ord=p) + onp.finfo(float).eps)
    h = onp.minimum(1, 1 / s)
    idx = y // h
    upper = (idx + 1) * h
    lower = idx * h
    prob = (upper - y) / (upper - lower + onp.finfo(float).eps)
    choice = onp.array(onp.random.uniform(size=len(x)) < prob, dtype='float')
    q = lower * choice + upper * (1. - choice)
    L = onp.int8(idx * choice + (idx + 1) * (1 - choice))
    Q = onp.linalg.norm(x, ord=p) * onp.sign(x) * q
    x_norm = onp.linalg.norm(x, ord=p)
    x_sign = onp.sign(x)

    bits = 32
    prev_nnz = -1
    code = []
    nnz_lst = []
    for i in range(x.size):
        if L[i] > 0:
            code.append(encode(L[i]))
            nnz_lst.append(encode(i - prev_nnz))
            bits += len(encode(i - prev_nnz)) + 1 + len(encode(L[i]))
            prev_nnz = i
    bits = int(onp.ceil(bits))

    return bits, code, nnz_lst, x_norm, x_sign, Q


def standard_dithering_plus(x, h, p):
    # h can be a scalar or a array with the same size as x
    y = onp.abs(x) / (onp.linalg.norm(x, ord=p) + onp.finfo(float).eps)
    idx = y // h
    upper = (idx + 1) * h
    lower = idx * h
    prob = (upper - y) / (upper - lower + onp.finfo(float).eps)
    choice = onp.array(onp.random.uniform(size=len(x)) < prob, dtype='float')
    q = lower * choice + upper * (1. - choice)
    L = onp.int8(idx * choice + (idx + 1) * (1 - choice))
    Q = onp.linalg.norm(x, ord=p) * onp.sign(x) * q
    x_norm = onp.linalg.norm(x, ord=p)
    x_sign = onp.sign(x)

    bits = 32
    prev_nnz = -1
    code = []
    nnz_lst = []
    for i in range(x.size):
        if L[i] > 0:
            code.append(encode(L[i]))
            nnz_lst.append(encode(i - prev_nnz))
            bits += len(encode(i - prev_nnz)) + 1 + len(encode(L[i]))
            prev_nnz = i
    bits = int(onp.ceil(bits))

    return bits, code, nnz_lst, x_norm, x_sign, Q


def run_gd(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'SD-DCGD.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_gd_plus(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'SD-DCGD-plus.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_gd_vnl(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'SD-DCGD-vnl.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_gd_plus_fnl(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'SD-DCGD-plus-fnl.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_diana(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'SD-DIANA.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_diana_plus(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'SD-DIANA-plus.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_diana_vnl(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'SD-DIANA-vnl.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_diana_plus_fnl(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'SD-DIANA-plus-fnl.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_sparse_diana_plus(algos_path, n_workers, it_max, dataset):
    file_name = algos_path + 'Sparse-DIANA-plus.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --data {3}".format(
            n_workers, file_name, it_max, dataset
        ))
    print('#', end='')


def run_block_diana(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'BL-DIANA.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_block_diana_plus(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'BL-DIANA-plus.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_block_dcgd_plus(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'BL-DCGD-plus.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_block_dcgd(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'BL-DCGD.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_block_diana_plus_fnl(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'BL-DIANA-plus-fnl.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_diag_diana_plus(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'diag-SD-DIANA-plus.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_diag_block_diana_plus(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'diag-BL-DIANA-plus.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')


def run_diag_block_diana_plus_safe(algos_path, n_workers, it_max, norm_type, dataset):
    file_name = algos_path + 'diag-SD-DIANA-plus-safe.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --norm_type {3} --data {4}".format(
            n_workers, file_name, it_max, norm_type, dataset
        ))
    print('#', end='')
