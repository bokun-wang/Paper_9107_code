from config import data_info, plots_path, is_convex, norm_type
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp
import pickle
import argparse
import os

sns.set(style="whitegrid", context="talk", palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = (6, 4)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

parser = argparse.ArgumentParser(description='Plotting the figures')
parser.add_argument('--algs', nargs='+', dest='algo_list', help='List of algorithms')
parser.add_argument('--data', action='store', dest='dataset', type=str, help='Dataset')
parser.add_argument('--n', action='store', dest='n_workers', type=int, help='Number of workers')
parser.add_argument('--goal', action='store', dest='goal', type=str, help='Goal')

args = parser.parse_args()
algo_list = args.algo_list
n_workers = args.n_workers
dataset = args.dataset
goal = args.goal
N, d = data_info[dataset]

if is_convex == True:
    problem_type = 'cvx'
else:
    problem_type = 'scvx'

all_dist_trace_ave = {}
all_time_trace_ave = {}
all_lower_dist_trace = {}
all_upper_dist_trace = {}
all_fgap_trace_ave = {}
all_lower_fgap_trace = {}
all_upper_fgap_trace = {}
all_max_val_trace_ave = {}
all_lower_max_val_trace = {}
all_upper_max_val_trace = {}
all_iter_trace_ave = {}
all_grad_oracles_trace_ave = {}
all_mbs_ave = {}

for alg in algo_list:
    save_path = '{0}/{1}-{2}-{3}-{4}-{5}'.format('./results', dataset, problem_type, norm_type, alg, str(n_workers))
    with open(save_path + '/dist_trace.p', "rb") as file:
        dist_trace = pickle.load(file)
    with open(save_path + '/fgap_trace.p', "rb") as file:
        fgap_trace = pickle.load(file)
    with open(save_path + '/max_value_trace.p', "rb") as file:
        max_val_trace = pickle.load(file)
    with open(save_path + '/iter_trace.p', "rb") as file:
        iter_trace = pickle.load(file)
    with open(save_path + '/grad_oracles_trace.p', "rb") as file:
        grad_oracles_trace = pickle.load(file)
    with open(save_path + '/time_trace.p', "rb") as file:
        time_trace = pickle.load(file)
    with open(save_path + '/mbs_trace.p', "rb") as file:
        mbs_trace = pickle.load(file)

    dist_trace_lst = list(dist_trace.values())
    fgap_trace_lst = list(fgap_trace.values())

    mbs_trace_lst = list(mbs_trace.values())
    mbs_trace_ave = onp.mean(mbs_trace_lst, axis=0)

    time_trace_lst = list(time_trace.values())
    time_trace_ave = onp.mean(time_trace_lst, axis=0)

    dist_trace_log = [onp.log(y) for y in dist_trace_lst]
    dist_trace_log_ave = onp.mean(dist_trace_log, axis=0)
    dist_trace_log_std = onp.std(dist_trace_log, axis=0)
    dist_trace_ave = onp.exp(dist_trace_log_ave)
    lower_dist_trace, upper_dist_trace = onp.exp(dist_trace_log_ave - dist_trace_log_std), onp.exp(
        dist_trace_log_ave + dist_trace_log_std)

    fgap_trace_log = [onp.log(y) for y in fgap_trace_lst]
    fgap_trace_log_ave = onp.mean(fgap_trace_log, axis=0)
    fgap_trace_log_std = onp.std(fgap_trace_log, axis=0)
    fgap_trace_ave = onp.exp(fgap_trace_log_ave)
    lower_fgap_trace, upper_fgap_trace = onp.exp(fgap_trace_log_ave - fgap_trace_log_std), onp.exp(
        fgap_trace_log_ave + fgap_trace_log_std)

    all_dist_trace_ave[alg] = dist_trace_ave
    all_lower_dist_trace[alg] = lower_dist_trace
    all_upper_dist_trace[alg] = upper_dist_trace

    all_fgap_trace_ave[alg] = fgap_trace_ave
    all_lower_fgap_trace[alg] = lower_fgap_trace
    all_upper_fgap_trace[alg] = upper_fgap_trace

    max_val_trace_lst = list(max_val_trace.values())
    max_val_trace_ave = onp.mean(max_val_trace_lst, axis=0)
    max_val_trace_std = onp.std(max_val_trace_lst, axis=0)

    lower_max_val_trace, upper_max_val_trace = (max_val_trace_ave - max_val_trace_std), (
            max_val_trace_ave + max_val_trace_std)

    all_max_val_trace_ave[alg] = max_val_trace_ave
    all_lower_max_val_trace[alg] = lower_max_val_trace
    all_upper_max_val_trace[alg] = upper_max_val_trace

    iter_trace_ave = onp.average(list(iter_trace.values()), axis=0)
    grad_oracles_trace_ave = onp.average(list(grad_oracles_trace.values()), axis=0)
    all_iter_trace_ave[alg] = iter_trace_ave
    all_grad_oracles_trace_ave[alg] = grad_oracles_trace_ave / N
    all_mbs_ave[alg] = mbs_trace_ave
    all_time_trace_ave[alg] = time_trace_ave


f1 = plt.figure()
colors = ['r', 'y', 'm', 'b', 'g', 'c']
markers = ['D', 'o', 'x', '*', 'v', '.']
vis_time = 10

for i, alg in enumerate(algo_list):

    grad_oracles_trace_ave = all_grad_oracles_trace_ave[alg]

    dist_trace_ave = all_dist_trace_ave[alg]
    lower_dist_trace = all_lower_dist_trace[alg]
    upper_dist_trace = all_upper_dist_trace[alg]
    iter_trace_ave = all_iter_trace_ave[alg]
    mbs_trace_ave = all_mbs_ave[alg]
    if goal == "nonuniform":
        if alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        elif alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'SD-DIANA-plus-fnl':
            legend_name = 'DIANA+ (quant)'
    elif goal == "improvement":
        if alg == 'SD-DCGD':
            legend_name = 'DCGD (quant)'
        elif alg == 'SD-DCGD-plus':
            legend_name = 'DCGD+ (quant+)'
        elif alg == 'SD-DCGD-plus-fnl':
            legend_name = 'DCGD+ (quant)'
        elif alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        elif alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'SD-DIANA-plus-fnl':
            legend_name = 'DIANA+ (quant)'
    elif goal == "neighborhood":
        if alg == 'SD-DCGD':
            legend_name = 'DCGD (quant)'
        elif alg == 'SD-DCGD-plus':
            legend_name = 'DCGD+ (quant+)'
    elif goal == "dithering_vs_sparse":
        if alg == 'Sparse-DIANA-plus':
            legend_name = 'DIANA+ (rand-'+r'$\tau$'+'+)'
        elif alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'BL-DIANA-plus':
            legend_name = 'DIANA+ (block quant+)'
    elif goal == "blockwise":
        if alg == 'BL-DIANA':
            legend_name = 'DIANA (block quant)'
        elif alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        elif alg == 'BL-DIANA-plus':
            legend_name = 'DIANA+ (block quant+)'
        elif alg == 'BL-DCGD':
            legend_name = 'DCGD (block quant)'
        elif alg == 'SD-DCGD':
            legend_name = 'DCGD (quant)'
        elif alg == 'BL-DCGD-plus':
            legend_name = 'DCGD+ (block quant+)'
        elif alg == 'BL-DIANA-plus-fnl':
            legend_name = 'DIANA+ (block quant)'
    elif goal == 'diagonal':
        if alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        if alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'BL-DIANA-plus':
            legend_name = 'DIANA+ (block quant+)'
        elif alg == 'diag-SD-DIANA-plus':
            legend_name = 'DIANA+ (diag, quant+)'
        elif alg == 'diag-SD-DIANA-plus-safe':
            legend_name = 'DIANA+ (safe diag, quant+)'
        elif alg == 'diag-BL-DIANA-plus':
            legend_name = 'DIANA+ (diag, block quant+)'
    plot = plt.plot(mbs_trace_ave, dist_trace_ave, markevery=max(1, len(grad_oracles_trace_ave) // vis_time),
                    marker=markers[i], label=legend_name, color=colors[i])
    plt.fill_between(mbs_trace_ave, lower_dist_trace, upper_dist_trace, alpha=0.25, color=colors[i])

plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.legend(prop={'size': 12})
plt.title('{0}, n={1}'.format(dataset, str(n_workers)))
plt.ylabel(r'$\frac{\Vert x^k-x^*\Vert^2}{\Vert x^0-x^*\Vert^2}$')
# plt.xticks([0.1, 0.5, 1.0])
plt.xlabel('Transmitted Megabytes')
fig_path = '{0}/{1}-{2}-{3}-{4}'.format(plots_path, dataset, problem_type, str(n_workers), norm_type)
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

plot_name1 = '-'.join(algo_list)
plt.savefig(fig_path + '/' + plot_name1 + '_dist_trace_mbs.pdf', dpi=300,
            bbox_inches='tight')


f2 = plt.figure()

for i, alg in enumerate(algo_list):

    grad_oracles_trace_ave = all_grad_oracles_trace_ave[alg]

    dist_trace_ave = all_dist_trace_ave[alg]
    lower_dist_trace = all_lower_dist_trace[alg]
    upper_dist_trace = all_upper_dist_trace[alg]
    iter_trace_ave = all_iter_trace_ave[alg]
    time_trace_ave = all_time_trace_ave[alg]
    if goal == "nonuniform":
        if alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        elif alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'SD-DIANA-plus-fnl':
            legend_name = 'DIANA+ (quant)'
    elif goal == "improvement":
        if alg == 'SD-DCGD':
            legend_name = 'DCGD (quant)'
        elif alg == 'SD-DCGD-plus':
            legend_name = 'DCGD+ (quant+)'
        elif alg == 'SD-DCGD-plus-fnl':
            legend_name = 'DCGD+ (quant)'
        elif alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        elif alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'SD-DIANA-plus-fnl':
            legend_name = 'DIANA+ (quant)'
    elif goal == "neighborhood":
        if alg == 'SD-DCGD':
            legend_name = 'DCGD (quant)'
        elif alg == 'SD-DCGD-plus':
            legend_name = 'DCGD+ (quant+)'
    elif goal == "dithering_vs_sparse":
        if alg == 'Sparse-DIANA-plus':
            legend_name = 'DIANA+ (rand-'+r'$\tau$'+'+)'
        elif alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'BL-DIANA-plus':
            legend_name = 'DIANA+ (block quant+)'
    elif goal == "blockwise":
        if alg == 'BL-DIANA':
            legend_name = 'DIANA (block quant)'
        elif alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        elif alg == 'BL-DIANA-plus':
            legend_name = 'DIANA+ (block quant+)'
        elif alg == 'BL-DCGD':
            legend_name = 'DCGD (block quant)'
        elif alg == 'SD-DCGD':
            legend_name = 'DCGD (quant)'
        elif alg == 'BL-DCGD-plus':
            legend_name = 'DCGD+ (block quant+)'
        elif alg == 'BL-DIANA-plus-fnl':
            legend_name = 'DIANA+ (block quant)'
    elif goal == 'diagonal':
        if alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        if alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'BL-DIANA-plus':
            legend_name = 'DIANA+ (block quant+)'
        elif alg == 'diag-SD-DIANA-plus':
            legend_name = 'DIANA+ (diag, quant+)'
        elif alg == 'diag-SD-DIANA-plus-safe':
            legend_name = 'DIANA+ (safe diag, quant+)'
        elif alg == 'diag-BL-DIANA-plus':
            legend_name = 'DIANA+ (diag, block quant+)'
    plot = plt.plot(time_trace_ave, dist_trace_ave, markevery=max(1, len(grad_oracles_trace_ave) // vis_time),
                    marker=markers[i], label=legend_name, color=colors[i])
    plt.fill_between(time_trace_ave, lower_dist_trace, upper_dist_trace, alpha=0.25, color=colors[i])

plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.legend(prop={'size': 12})
plt.title('{0}, n={1}'.format(dataset, str(n_workers)))
plt.ylabel(r'$\frac{\Vert x^k-x^*\Vert^2}{\Vert x^0-x^*\Vert^2}$')
plt.xlabel('Time (seconds)')
fig_path = '{0}/{1}-{2}-{3}-{4}'.format(plots_path, dataset, problem_type, str(n_workers), norm_type)
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

plot_name1 = '-'.join(algo_list)
plt.savefig(fig_path + '/' + plot_name1 + '_dist_trace_time.pdf', dpi=300,
            bbox_inches='tight')

f3 = plt.figure()
colors = ['r', 'y', 'm', 'b', 'g', 'c']
markers = ['D', 'o', 'x', '*', 'v', '.']
vis_time = 10

for i, alg in enumerate(algo_list):

    grad_oracles_trace_ave = all_grad_oracles_trace_ave[alg]

    dist_trace_ave = all_dist_trace_ave[alg]
    lower_dist_trace = all_lower_dist_trace[alg]
    upper_dist_trace = all_upper_dist_trace[alg]
    iter_trace_ave = all_iter_trace_ave[alg]
    mbs_trace_ave = all_mbs_ave[alg]
    if goal == "nonuniform":
        if alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        elif alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'SD-DIANA-plus-fnl':
            legend_name = 'DIANA+ (quant)'
    elif goal == "improvement":
        if alg == 'SD-DCGD':
            legend_name = 'DCGD (quant)'
        elif alg == 'SD-DCGD-plus':
            legend_name = 'DCGD+ (quant+)'
        elif alg == 'SD-DCGD-plus-fnl':
            legend_name = 'DCGD+ (quant)'
        elif alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        elif alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'SD-DIANA-plus-fnl':
            legend_name = 'DIANA+ (quant)'
    elif goal == "neighborhood":
        if alg == 'SD-DCGD':
            legend_name = 'DCGD (quant)'
        elif alg == 'SD-DCGD-plus':
            legend_name = 'DCGD+ (quant+)'
    elif goal == "dithering_vs_sparse":
        if alg == 'Sparse-DIANA-plus':
            legend_name = 'DIANA+ (rand-'+r'$\tau$'+'+)'
        elif alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'BL-DIANA-plus':
            legend_name = 'DIANA+ (block quant+)'
    elif goal == "blockwise":
        if alg == 'BL-DIANA':
            legend_name = 'DIANA (block quant)'
        elif alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        elif alg == 'BL-DIANA-plus':
            legend_name = 'DIANA+ (block quant+)'
        elif alg == 'BL-DCGD':
            legend_name = 'DCGD (block quant)'
        elif alg == 'SD-DCGD':
            legend_name = 'DCGD (quant)'
        elif alg == 'BL-DCGD-plus':
            legend_name = 'DCGD+ (block quant+)'
        elif alg == 'BL-DIANA-plus-fnl':
            legend_name = 'DIANA+ (block quant)'
    elif goal == 'diagonal':
        if alg == 'SD-DIANA':
            legend_name = 'DIANA (quant)'
        if alg == 'SD-DIANA-plus':
            legend_name = 'DIANA+ (quant+)'
        elif alg == 'BL-DIANA-plus':
            legend_name = 'DIANA+ (block quant+)'
        elif alg == 'diag-SD-DIANA-plus':
            legend_name = 'DIANA+ (diag, quant+)'
        elif alg == 'diag-SD-DIANA-plus-safe':
            legend_name = 'DIANA+ (safe diag, quant+)'
        elif alg == 'diag-BL-DIANA-plus':
            legend_name = 'DIANA+ (diag, block quant+)'
    plot = plt.plot(iter_trace_ave, dist_trace_ave, markevery=max(1, len(grad_oracles_trace_ave) // vis_time),
                    marker=markers[i], label=legend_name, color=colors[i])
    plt.fill_between(iter_trace_ave, lower_dist_trace, upper_dist_trace, alpha=0.25, color=colors[i])

plt.yscale('log')
plt.tight_layout()
plt.legend(prop={'size': 12})
plt.title('{0}, n={1}'.format(dataset, str(n_workers)))
plt.ylabel(r'$\frac{\Vert x^k-x^*\Vert^2}{\Vert x^0-x^*\Vert^2}$')
plt.xlabel('Iterations')
fig_path = '{0}/{1}-{2}-{3}-{4}'.format(plots_path, dataset, problem_type, str(n_workers), norm_type)
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

plot_name1 = '-'.join(algo_list)
plt.savefig(fig_path + '/' + plot_name1 + '_dist_trace_iters.pdf', dpi=300,
            bbox_inches='tight')
