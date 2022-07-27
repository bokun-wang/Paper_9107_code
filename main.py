from config import data_info, algos_path, default_it_max, norm_type
from utils import run_gd, run_gd_plus, run_gd_plus_fnl, run_diana, run_diana_plus, run_diana_vnl, run_diana_plus_fnl, \
    run_gd_vnl, run_sparse_diana_plus, run_block_diana, run_block_diana_plus, run_block_dcgd, run_block_dcgd_plus, \
    run_block_diana_plus_fnl, run_diag_diana_plus, run_diag_block_diana_plus, run_diag_block_diana_plus_safe
import argparse

parser = argparse.ArgumentParser(description='Run the Algorithms')
parser.add_argument('--data', action='store', dest='dataset', type=str, help='Dataset')
parser.add_argument('--alg', action='store', dest='algo_name', type=str,
                    help='Which algorithm: SD-DCGD')
parser.add_argument('--n', action='store', dest='n_workers', type=int, help='Number of workers')

args = parser.parse_args()
dataset = args.dataset
algo_name = args.algo_name
n_workers = args.n_workers

N, d = data_info[dataset]

if algo_name == 'SD-DCGD':
    it_max = int(default_it_max)
    run_gd(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'SD-DCGD-plus':
    it_max = int(default_it_max)
    run_gd_plus(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'SD-DCGD-plus-fnl':
    it_max = int(default_it_max)
    run_gd_plus_fnl(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'SD-DCGD-vnl':
    it_max = int(default_it_max)
    run_gd_vnl(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'SD-DIANA':
    it_max = int(default_it_max)
    run_diana(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'SD-DIANA-plus':
    it_max = int(default_it_max)
    run_diana_plus(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'SD-DIANA-plus-fnl':
    it_max = int(default_it_max)
    run_diana_plus_fnl(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'BL-DIANA':
    it_max = int(default_it_max)
    run_block_diana(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'BL-DCGD':
    it_max = int(default_it_max)
    run_block_dcgd(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'BL-DCGD-plus':
    it_max = int(default_it_max)
    run_block_dcgd_plus(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'BL-DIANA-plus':
    it_max = int(default_it_max)
    run_block_diana_plus(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'BL-DIANA-plus-fnl':
    it_max = int(default_it_max)
    run_block_diana_plus_fnl(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'diag-SD-DIANA-plus':
    it_max = int(default_it_max)
    run_diag_diana_plus(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'diag-SD-DIANA-plus-safe':
    it_max = int(default_it_max)
    run_diag_block_diana_plus_safe(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'diag-BL-DIANA-plus':
    it_max = int(default_it_max)
    run_diag_block_diana_plus(algos_path, n_workers, it_max, norm_type, dataset)
elif algo_name == 'Sparse-DIANA-plus':
    it_max = int(default_it_max)
    run_sparse_diana_plus(algos_path, n_workers, it_max, dataset)
else:
    raise ValueError('The algorithm has not been implemented!')
