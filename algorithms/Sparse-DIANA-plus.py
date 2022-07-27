import os, sys
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass
from loss_functions.logistic_regression import LogisticRegression
from helpers.lr_handler import LR_Handler
from helpers.trace_handler import Trace_Handler
from utils import set_seed, ind_sparsification
import numpy as onp
import argparse
from mpi4py import MPI
from config import data_path, eval_ratio, l2_weights, lr_decay, seeds, is_convex
import pickle
import glob

import scipy.optimize

from my_timer import Timer

parser = argparse.ArgumentParser(description='Run DIANA+ (sparsification)')
parser.add_argument('--it', action='store', dest='it_max', type=int, help='Numer of Iterations')
parser.add_argument('--data', action='store', dest='dataset', type=str, help='Dataset')

args = parser.parse_args()
it_max = args.it_max
dataset = args.dataset

if is_convex == True:
    l2 = 0
    problem_type = 'cvx'
else:
    l2 = l2_weights[dataset]
    problem_type = 'scvx'

trace_period = onp.floor(it_max * eval_ratio)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size()

timer = Timer()

# Load local data
file_name = '{0}{1}-{2}-{3}/{4}'.format(data_path, dataset, problem_type, n_workers, rank)
with open(file_name + '/A.p', "rb") as file:
    Ai = pickle.load(file)
with open(file_name + '/b.p', "rb") as file:
    bi = pickle.load(file)

m = bi.shape[0]
loss_i = LogisticRegression(Ai, bi, l2=l2)

# initial point
x0 = onp.zeros((Ai.shape[1],))
h0 = onp.zeros((Ai.shape[1],))
fi_0 = loss_i.value(x0)
f_0 = comm.allreduce(fi_0, op=MPI.SUM) / n_workers

##############################################################################
dim = loss_i.dim
tau = onp.floor(dim / n_workers)
##############################################################################

# load opt sol for eval
if rank == 0:
    opt_file_path = '{0}{1}-{2}-{3}'.format(data_path, dataset, problem_type, n_workers)
    abs_path = glob.glob(opt_file_path + '/*_info.p')
    with open(abs_path[0], "rb") as file:
        opt_info = pickle.load(file)

    x_opt = opt_info['x_opt']
    f_opt = opt_info['f_opt']
    L = opt_info['L']
    r_0_func = f_0 - f_opt
    r_0_dist = onp.linalg.norm(x0 - x_opt) ** 2

    trace_handler = Trace_Handler(all_seeds=seeds[0], problem_type=problem_type, x_opt=x_opt, f_opt=f_opt)
else:
    L = None
    r_0_func = None
    r_0_dist = None
    trace_handler = None

L = comm.bcast(L, root=0)
Li = onp.array(loss_i.matrix_smoothness())
diag_Li = onp.diag(Li)
ui, si, vhi = onp.linalg.svd(Li, full_matrices=False)
si_sqrt = onp.sqrt(si)
si_isqrt = onp.sqrt(1 / si)

mLi_sqrt = onp.dot(ui * si_sqrt, vhi)
mLi_isqrt = onp.dot(ui * si_isqrt, vhi)

##### Calculate the optimal probabilities
diag_Li_prm = diag_Li / (n_workers * l2) + 1
Func_p = lambda x: onp.sum(diag_Li_prm / (diag_Li_prm + x)) - tau
sol = scipy.optimize.root(Func_p, 1)
rho_prm = sol.x
# rho_prm = scipy.optimize.broyden1(Func_p, 1, f_tol=1e-14)
prob_i = diag_Li_prm / (diag_Li_prm + rho_prm)

omega_i = onp.max(1/prob_i - 1)
omega_max = comm.allreduce(omega_i, op=MPI.MAX)

tilde_L_i = onp.max((1/prob_i - 1) * diag_Li)
tilde_L_max = comm.allreduce(tilde_L_i, op=MPI.MAX)

if problem_type == 'cvx':
    lr0 = 1 / (2 * (L + 6 * tilde_L_max / n_workers))
else:
    lr0 = 1 / (L + 6 * tilde_L_max / n_workers)

if rank == 0:
    lr_handler = LR_Handler(num_iters=it_max, decay=lr_decay, lr0=lr0)
    all_mbs = {}
    all_time = {}
else:
    lr_handler = None
    all_mbs = None
    all_time = None

seeds_i = seeds[rank]

# Initialization
comm.Barrier()
start = MPI.Wtime()
all_mLi_sqrt = comm.gather(mLi_sqrt, root=0)
comm.Barrier()
end = MPI.Wtime()
time_init = (end - start)

for seed in seeds_i:
    # set the random seed
    # seed_i = seed  # + rank  # different seeds for different workers
    set_seed(seed)
    x = onp.copy(x0)
    hi_k = onp.copy(h0)
    h_k = onp.copy(h0)
    if rank == 0:
        lr_handler.set_lr()

    n_bits = dim * dim * 32 * n_workers
    n_grad = 0
    total_time = time_init
    for k in range(it_max):

        if k % trace_period == 0:
            fi_k = loss_i.value(x)
            f_k = comm.allreduce(fi_k, op=MPI.SUM) / n_workers

            if rank == 0:
                trace_handler.update_trace(seed=seed, x=x, f_eval=f_k, r_0_dist=r_0_dist, r_0_func=r_0_func)
                trace_handler.update_ngrads(n_grads=n_workers * m * trace_period, seed=seed, iter=k)
                trace_handler.update_mbs(n_bits=n_bits, seed=seed)
                trace_handler.update_time(time=total_time, seed=seed)

        if rank == 0:
            lr = lr_handler.get_lr()
        else:
            lr = None

        lr = comm.bcast(lr, root=0)
        timer.start()
        local_g = loss_i.gradient(x)
        time_compute_grad = timer.stop()
        if rank == 0:
            total_time += time_compute_grad

        timer.start()
        sparsed = ind_sparsification(onp.matmul(mLi_isqrt, local_g - hi_k), p=prob_i)
        n_bit = tau * 32
        ################################
        Qi_k = onp.matmul(mLi_sqrt, sparsed)
        ################################
        time_sparsification = timer.stop()
        if rank == 0:
            total_time += time_sparsification

        timer.start()
        hi_k_next = hi_k + Qi_k / (omega_max + 1)
        hi_k = onp.copy(hi_k_next)
        time_update_shift = timer.stop()
        if rank == 0:
            total_time += time_update_shift

        n_bit_all = comm.allreduce(n_bit, op=MPI.SUM)
        n_bits += n_bit_all

        comm.Barrier()
        start = MPI.Wtime()
        all_sparsed = comm.gather(sparsed, root=0)

        comm.Barrier()
        end = MPI.Wtime()
        if rank == 0:
            total_time += (end - start)

        timer.start()
        if rank == 0:
            Q_k = onp.zeros_like(x)
            for i in range(n_workers):
                sparsed_i = all_sparsed[i]
                mLi_sqrt_i = all_mLi_sqrt[i]
                Q_k += onp.matmul(mLi_sqrt_i, sparsed_i)
        else:
            Q_k = None
        time_decoding = timer.stop()
        if rank == 0:
            total_time += time_decoding

        comm.Barrier()
        start = MPI.Wtime()
        Q_k = comm.bcast(Q_k, root=0)
        comm.Barrier()
        end = MPI.Wtime()
        if rank == 0:
            total_time += (end - start)

        timer.start()
        g_k = h_k + Q_k / n_workers
        x = x - lr * g_k
        time_grad_step = timer.stop()
        if rank == 0:
            total_time += time_grad_step

        timer.start()
        h_k_next = h_k + Q_k / (n_workers * (omega_max + 1))
        h_k = onp.copy(h_k_next)

        time_glob_shift = timer.stop()
        if rank == 0:
            total_time += time_glob_shift

print("Rank %d is down" % rank)

if rank == 0:
    time_trace, grad_oracles_trace, dist_trace, fgap_trace, max_value_trace, _, iter_trace, mbs_trace = trace_handler.get_trace()
    algo_name = 'Sparse-DIANA-plus'
    norm_type = "inf"
    save_path = '{0}/{1}-{2}-{3}-{4}-{5}'.format('./results', dataset, problem_type, norm_type, algo_name, n_workers)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pickle.dump(time_trace, open(save_path + "/time_trace.p", "wb"))
    pickle.dump(grad_oracles_trace, open(save_path + "/grad_oracles_trace.p", "wb"))
    pickle.dump(dist_trace, open(save_path + "/dist_trace.p", "wb"))
    pickle.dump(fgap_trace, open(save_path + "/fgap_trace.p", "wb"))
    pickle.dump(max_value_trace, open(save_path + "/max_value_trace.p", "wb"))
    pickle.dump(iter_trace, open(save_path + "/iter_trace.p", "wb"))
    pickle.dump(mbs_trace, open(save_path + "/mbs_trace.p", "wb"))
