import itertools
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from collections import defaultdict

use_default_cm = False

# colors = "bgrycmkwbgrycmkw"
colors = ('blue', 'orange', 'green', 'red', 'cyan',
          'brown', 'pink', 'olive', 'gray', 'purple',
          'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray')
if use_default_cm:
    color_map = {}
else:
    color_map = {
        'ours ': 'blue',
        'equ': 'blue',
        '8x RAD FC-GQ-CNN': 'red',
        '8x RAD VPG': 'green',
        '16 soft equ FC-GQ-CNN': 'brown',
        '4x soft equ VPG': 'yellow',
        '8x RAD FC-GQ-CNN': 'orange',
        'fcgqcnn': 'orange',
        '8x RAD VPG': 'purple',
        'vpg': 'purple',
        'no opt': 'red',
        'rot equ': 'green',
        'no ASR': 'orange',
        'no equ': 'purple',
        'ggcnn': 'green',
        'grconvnet': 'red'
    }

linestyle_map = {
}
name_map = {
    'equ': 'ours ',
    'fcgqcnn': 'FC-GQ-CNN',
    'vpg': 'VPG',
    'ggcnn': 'GG-CNN',
    'grconvnet': 'GR-Conv-Net',
}

sequence = {
    '8x RAD VPG': 1,
    'vpg': 1,
    'no opt': 1.6,
    'Rot FCN': 2,
    'rot equ': 2.05,
    '2x RAD VPG': 2.21,
    '4x RAD VPG': 2.22,
    '8x RAD VPG': 2.23,
    '2x soft equ VPG': 2.31,
    '4x soft equ VPG': 2.32,
    '8x soft equ VPG': 2.33,
    '8x RAD FC-GQ-CNN': 2.4,
    'fcgqcnn': 2.4,
    '2 soft equ FC-GQ-CNN': 2.51,
    '4 soft equ FC-GQ-CNN': 2.52,
    '8 soft equ FC-GQ-CNN': 2.53,
    '16 soft equ FC-GQ-CNN': 2.54,
    '2x RAD FC-GQ-CNN': 2.61,
    '4x RAD FC-GQ-CNN': 2.62,
    '8x RAD FC-GQ-CNN': 2.63,
    'ggcnn':2.7,
    'no ASR': 2.8,
    'no equ': 3,
    'ASR': 3.1,
    'FCN': 4,
    'DrQ ASR': 5,
    'RAD ASR': 6,
    '8RAD ASR': 6,
    'RAD FCN': 7,
    '4RAD FCN': 7,
    'RAD Rot FCN': 8,
    '4RAD Rot FCN': 8,
    'grconvnet': 8.7,
    'equ': 9,
    'ours ': 9,
}

def getRewardsSingle(rewards, window=1000):
    moving_avg = []
    i = window
    # while i - window < len(rewards):
    #     moving_avg.append(np.average(rewards[i - window:i]))
    #     i += window
    multi_window_len = (len(rewards) // window) * window
    rewards = np.array(rewards[:multi_window_len])
    moving_avg = rewards.reshape(-1, window).mean(1)

    return moving_avg


def plotLearningCurveAvg(rewards, window=1000, label='reward', color='b', shadow=True, ax=plt, legend=True,
                         linestyle='-', show_success_rate=False):
    lens = list(len(i) for i in rewards)
    min_len = min(lens)
    max_len = max(lens)
    if min_len != max_len:
        rewards = np.array(list(itertools.zip_longest(*rewards, fillvalue=0))).T
        rewards = rewards[:, :min_len]
    avg_rewards = np.mean(rewards, axis=0)
    # std_rewards = np.std(rewards, axis=0)
    std_rewards = stats.sem(rewards, axis=0)
    xs = np.arange(window, window * avg_rewards.shape[0] + 1, window)
    if shadow:
        ax.fill_between(xs, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, color=color)
    l = ax.plot(xs, avg_rewards, label=label, color=color, linestyle=linestyle, alpha=0.7)
    if legend:
        ax.legend(loc=4)
    if show_success_rate:
        print(label, 'step 450: {:.3f}+-{:.3f} step 1500: {:.3f}+-{:.3f}'.format
        (avg_rewards[2], std_rewards[2], avg_rewards[9], std_rewards[9]))
    return l


def plotEvalCurveAvg(sr, datasize, label='reward', color='b', shadow=True, ax=plt, linestyle='-'):
    avg_sr = np.mean(sr, axis=0)
    std_sr = stats.sem(sr, axis=0)
    if shadow:
        ax.fill_between(datasize, avg_sr - std_sr, avg_sr + std_sr, alpha=0.2, color=color)
    l = ax.plot(datasize, avg_sr, label=label, color=color, linestyle=linestyle, alpha=0.7)
    return l


def plotEvalBarAvg(sr, datasize, label='reward', color='b', shadow=True, ax=plt, linestyle='-', idx=0):
    avg_sr = np.mean(sr, axis=0)
    avg_sr[avg_sr < 0.005] = 0.005
    std_sr = stats.sem(sr, axis=0)
    x = np.arange(len(datasize)) + idx * 0.12 - 2 * 0.12
    bar_width = 0.1
    # if shadow:
    #     ax.fill_between(datasize, avg_sr - std_sr, avg_sr + std_sr, alpha=0.2, color=color)
    # l = ax.plot(datasize, avg_sr, label=label, color=color, linestyle=linestyle, alpha=0.7)
    l = ax.bar(x, avg_sr, yerr=std_sr, color=color, width=bar_width, label=label)
    return l


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def plotEvalRS_Curve_Datasize(base, filer_pass_word='_', figname='plot.png'):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    LEGEND_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    methods = defaultdict(lambda: [])
    # methods: {'method1': [(datasize1, SRs), (datasize2, SRs)],
    #           ...,
    #           'method3': [(datasize1, SRs), (datasize2, SRs)]}
    method_datasizes = filter(lambda x: (x[0] != '.' and x.find(filer_pass_word) >= 0),
                              get_immediate_subdirectories(base))
    # method_datasizes = filter(lambda x: (x[0] != '.'), get_immediate_subdirectories(base))
    for method_datasize in sorted(method_datasizes, key=lambda x: str(sequence[x]) if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method_datasize))):
            try:
                r = np.load(os.path.join(base, method_datasize, run, 'info/eval_rewards.npy'))
                if method_datasize.find('_BC_') >= 0:
                    rs.append(r)
                else:
                    rs.append(r.max())
            except Exception as e:
                continue
        SRs = np.asarray(rs).reshape(1, -1)
        method = method_datasize[:method_datasize.find('_')]
        datasize = method_datasize[method_datasize.rfind('_') + 1:]
        data = np.insert(SRs, 0, datasize)
        methods[method].append(data.T)

    i = 0
    for method, results in methods.items():
        results = np.asarray(results).T
        SRs = results[1:, :]
        datasizes = results[:1, :].reshape(-1)
        idx = np.argsort(datasizes)
        SRs = SRs[:, idx]
        datasizes = datasizes[idx]
        plotEvalCurveAvg(SRs, datasizes,
                         label=name_map[method]
                         if method in name_map else method,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else ':')
        i += 1

    plt.legend(loc=4, facecolor='w', fontsize=LEGEND_SIZE, framealpha=0.6)
    plt.xscale('log', base=2)
    plt.xlabel('Dataset size')
    plt.ylabel('Validation SR')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(base, figname + 'SL_validation'), bbox_inches='tight', pad_inches=0)


def plotEvalRS_Bar_Datasize(base, filer_pass_word='_', figname='plot.png'):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    LEGEND_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    methods = defaultdict(lambda: [])
    # methods: {'method1': [(datasize1, SRs), (datasize2, SRs)],
    #           ...,
    #           'method3': [(datasize1, SRs), (datasize2, SRs)]}
    method_datasizes = filter(lambda x: (x[0] != '.' and x.find(filer_pass_word) >= 0),
                              get_immediate_subdirectories(base))
    # method_datasizes = filter(lambda x: (x[0] != '.'), get_immediate_subdirectories(base))
    for method_datasize in sorted(method_datasizes, key=lambda x: str(sequence[x]) if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method_datasize))):
            try:
                r = np.load(os.path.join(base, method_datasize, run, 'info/eval_rewards.npy'))
                if method_datasize.find('_BC_') >= 0:
                    rs.append(r)
                else:
                    rs.append(r.max())
            except Exception as e:
                continue
        SRs = np.asarray(rs).reshape(1, -1)
        method = method_datasize[:method_datasize.find('_')]
        datasize = method_datasize[method_datasize.rfind('_') + 1:]
        data = np.insert(SRs, 0, datasize)
        methods[method].append(data.T)

    i = 0
    for method in sorted(methods, key=lambda x: str(sequence[x]) if x in sequence.keys() else x):
        results = methods[method]
        results = np.asarray(results).T
        SRs = results[1:, :]
        datasizes = results[:1, :].reshape(-1)
        idx = np.argsort(datasizes)
        SRs = SRs[:, idx]
        datasizes = datasizes[idx]
        plotEvalBarAvg(SRs, datasizes,
                       label=name_map[method]
                       if method in name_map else method,
                       color=color_map[method] if method in color_map else colors[i],
                       linestyle=linestyle_map[method] if method in linestyle_map else ':',
                       idx=i)
        i += 1

    plt.legend(loc=4, facecolor='w', fontsize=LEGEND_SIZE, framealpha=0.6)
    # plt.xscale('log', base=2)
    plt.xlim(-0.35, 5.5)
    plt.xlabel('Dataset size')
    plt.ylabel('Validation SR')
    plt.ylim(-0.05, 1)
    plt.xticks(np.arange(len(datasizes)), [str(int(i)) for i in datasizes])
    plt.tight_layout()
    plt.savefig(os.path.join(base, figname + 'SL_validation'), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    folder = 'supervised learning'
    base = '/home/zxp-s-works/grasp_data/' \
           + folder
    # plotEvalRS_Curve_Datasize(base, filer_pass_word='_', figname=folder)
    plotEvalRS_Bar_Datasize(base, filer_pass_word='_', figname=folder)
