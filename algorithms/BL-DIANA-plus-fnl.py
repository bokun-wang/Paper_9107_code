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
from utils import set_seed, standard_dithering_plus
import numpy as onp
import argparse
from mpi4py import MPI
from config import data_path, eval_ratio, l2_weights, lr_decay, seeds, is_convex
import pickle
import glob

from my_timer import Timer

from EliasOmega import decode

parser = argparse.ArgumentParser(description='Run DIANA+ (block quant, fnl)')
parser.add_argument('--it', action='store', dest='it_max', type=int, help='Numer of Iterations')
parser.add_argument('--data', action='store', dest='dataset', type=str, help='Dataset')
parser.add_argument('--norm_type', action='store', dest='norm_type', type=float, help='Type of norm')

args = parser.parse_args()
it_max = args.it_max
dataset = args.dataset
norm_type = args.norm_type

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

###############################

dim = loss_i.dim
n_blocks = n_workers
dim_b = dim // n_blocks

beta = dim / n_blocks + n_blocks
h_final = (onp.sqrt(dim_b) + onp.sqrt(dim_b + 4 * (beta / n_blocks - 1))) / (2 * (beta / n_blocks - 1))
###############################
nu = onp.minimum(norm_type, 2)

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
    trace_handler = None

L = comm.bcast(L, root=0)
Li = onp.array(loss_i.matrix_smoothness())
diag_Li = onp.diag(Li)
ui, si, vhi = onp.linalg.svd(Li, full_matrices=False)
si_sqrt = onp.sqrt(si)
si_isqrt = onp.sqrt(1 / si)

mLi_sqrt = onp.dot(ui * si_sqrt, vhi)
mLi_isqrt = onp.dot(ui * si_isqrt, vhi)

all_dl_sqrt = onp.zeros(n_blocks)
all_norm_il = onp.zeros(n_blocks)
for bl in range(n_blocks):
    if bl < n_blocks - 1:
        coord_idx = onp.arange(bl * dim_b, (bl + 1) * dim_b)
    else:
        coord_idx = onp.arange(bl * dim_b, dim)
    all_dl_sqrt[bl] = onp.sqrt(len(coord_idx))
    all_norm_il[bl] = onp.linalg.norm(diag_Li[coord_idx], ord=2)

omega_sd = onp.max(h_final * all_dl_sqrt)
omega_sd_max = comm.allreduce(omega_sd, op=MPI.MAX)

tilde_L_i = onp.max(h_final * all_norm_il)
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
all_h_final = comm.gather(h_final, root=0)
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

    n_bits = (dim * dim + n_blocks) * 32 * n_workers
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
        all_block_nbit = 0
        all_block_code = []
        all_block_nnz_lst = []
        all_block_x_norm = []
        all_block_x_sign = []
        Qi_k_ = onp.zeros_like(x)
        Precond_g = onp.matmul(mLi_isqrt, local_g - hi_k)
        for bl in range(n_blocks):
            if bl < n_blocks - 1:
                coord_idx = onp.arange(bl * dim_b, (bl + 1) * dim_b)
            else:
                coord_idx = onp.arange(bl * dim_b, dim)
            Precond_g_bl = Precond_g[coord_idx]
            h_final_bl = h_final
            n_bit_bl, code_bl, nnz_lst_bl, x_norm_bl, x_sign_bl, Qi_k_bl = standard_dithering_plus(Precond_g_bl, h_final_bl,
                                                                                              norm_type)
            all_block_nbit += n_bit_bl
            all_block_code.append(code_bl)
            all_block_nnz_lst.append(nnz_lst_bl)
            all_block_x_norm.append(x_norm_bl)
            all_block_x_sign.append(x_sign_bl)
            Qi_k_[coord_idx] = Qi_k_bl

        ################################
        Qi_k = onp.matmul(mLi_sqrt, Qi_k_)
        ################################

        time_dithering = timer.stop()
        if rank == 0:
            total_time += time_dithering

        timer.start()
        hi_k_next = hi_k + Qi_k / (omega_sd_max + 1)
        hi_k = onp.copy(hi_k_next)
        time_update_shift = timer.stop()
        if rank == 0:
            total_time += time_update_shift

        n_bit_all = comm.allreduce(all_block_nbit, op=MPI.SUM)
        n_bits += n_bit_all

        comm.Barrier()
        start = MPI.Wtime()
        all_codes = comm.gather(all_block_code, root=0)
        all_norms = comm.gather(all_block_x_norm, root=0)
        all_signs = comm.gather(all_block_x_sign, root=0)
        all_nnz_lsts = comm.gather(all_block_nnz_lst, root=0)
        comm.Barrier()
        end = MPI.Wtime()
        if rank == 0:
            total_time += (end - start)

        timer.start()
        if rank == 0:
            Q_k = onp.zeros_like(x)
            for i in range(n_workers):
                Q_k_i = onp.zeros_like(x)
                mLi_sqrt_i = all_mLi_sqrt[i]
                for bl in range(n_blocks):
                    if bl < n_blocks - 1:
                        coord_idx = onp.arange(bl * dim_b, (bl + 1) * dim_b)
                    else:
                        coord_idx = onp.arange(bl * dim_b, dim)
                    code_i_bl = all_codes[i][bl]
                    norm_i_bl = all_norms[i][bl]
                    sign_i_bl = all_signs[i][bl]
                    h_i_bl = all_h_final[i]
                    nnz_lst_i_bl = all_nnz_lsts[i][bl]
                    ell_i_bl = onp.zeros_like(x[coord_idx])
                    prev_nnz = -1
                    for j in range(len(code_i_bl)):
                        idx = decode(nnz_lst_i_bl[j])[0] + prev_nnz
                        ell_i_bl[idx] = decode(code_i_bl[j])[0]
                        prev_nnz = idx

                    Q_k_i[coord_idx] = norm_i_bl * sign_i_bl * (ell_i_bl * h_i_bl)

                Q_k += onp.matmul(mLi_sqrt_i, Q_k_i)
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
        h_k_next = h_k + Q_k / (n_workers * (omega_sd_max + 1))
        h_k = onp.copy(h_k_next)

        time_glob_shift = timer.stop()
        if rank == 0:
            total_time += time_glob_shift

print("Rank %d is down" % rank)

if rank == 0:
    time_trace, grad_oracles_trace, dist_trace, fgap_trace, max_value_trace, _, iter_trace, mbs_trace = trace_handler.get_trace()
    algo_name = 'BL-DIANA-plus-fnl'
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
