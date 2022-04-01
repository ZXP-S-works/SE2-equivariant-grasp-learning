import itertools
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

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
        'FC-GQ-CNN': 'orange',
        'FC-GQ-CNN ': 'orange',
        '8x RAD FC-GQ-CNN': 'red',
        '4x RAD FC-GQ-CNN': 'green',
        '2x RAD FC-GQ-CNN': 'purple',
        '8x RAD VPG': 'green',
        '4x RAD VPG': 'red',
        '2x RAD VPG': 'orange',
        '16 soft equ FC-GQ-CNN': 'brown',
        '16x soft equ FC-GQ-CNN': 'brown',
        '8x soft equ FC-GQ-CNN': 'red',
        '4x soft equ FC-GQ-CNN': 'green',
        '2x soft equ FC-GQ-CNN': 'purple',
        '8x soft equ VPG': 'red',
        '4x soft equ VPG': 'orange',
        '2x soft equ VPG': 'green',
        'VPG': 'purple',
        'VPG ': 'purple',
        'no opt': 'red',
        '8x RAD rot FCN': 'red',
        'rot equ': 'green',
        '4x RAD rot FCN': 'orange',
        '2x RAD rot FCN': 'purple',
        'rot FCN': 'brown',
        'no ASR': 'orange',
        'no equ': 'purple',
    }

linestyle_map = {
}

name_map = {
}

sequence = {
        'ours ': 1,
        '8x RAD FC-GQ-CNN': 2.1,
        '4x RAD FC-GQ-CNN': 2.2,
        '2x RAD FC-GQ-CNN': 2.3,
        '8x RAD VPG': 3.1,
        '4x RAD VPG': 3.2,
        '2x RAD VPG': 3.3,
        '16x soft equ FC-GQ-CNN': 4.1,
        '8x soft equ FC-GQ-CNN': 4.2,
        '4x soft equ FC-GQ-CNN': 4.3,
        '2x soft equ FC-GQ-CNN': 4.4,
        '8x soft equ VPG': 5.1,
        '4x soft equ VPG': 5.2,
        '2x soft equ VPG': 5.3,
        'FC-GQ-CNN': 6,
        'VPG': 7,
        'no opt': 8,
        '8x RAD rot FCN': 9,
        'rot equ': 9.1,
        '4x RAD rot FCN': 9.2,
        '2x RAD rot FCN': 9.3,
        'rot FCN': 9.4,
        'no ASR': 10,
        'no equ': 11,
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


def plotEvalCurveAvg(rewards, freq=1000, label='reward', color='b', shadow=True, ax=plt, legend=True,
                     linestyle='-', start_len=0, end_len=None):
    lens = list(len(i) for i in rewards)
    min_len = min(lens)
    max_len = max(lens)
    start_len = max(min(start_len, min_len * freq), freq)
    end_len = min(end_len, min_len * freq) if end_len is not None else (min_len + 1) * freq
    start_idx = int(start_len / freq) - 1
    end_idx = int(end_len / freq)
    rewards = np.array(list(itertools.zip_longest(*rewards, fillvalue=0))).T
    rewards = rewards[:, start_idx:end_idx]
    avg_rewards = np.mean(rewards, axis=0)
    # std_rewards = np.std(rewards, axis=0)
    std_rewards = stats.sem(rewards, axis=0)
    pltx_start = freq * start_idx + freq
    pltx_end = freq * end_idx + 1
    xs = np.arange(pltx_start, pltx_end, freq)
    if shadow:
        ax.fill_between(xs, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, color=color)
    l = ax.plot(xs, avg_rewards, label=label, color=color, linestyle=linestyle, alpha=0.7)
    # if legend:
    #     ax.legend(loc=4)
    return l


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def plotLearningCurve(base, ep=50000, use_default_cm=False, filer_pass_word='_', figname='plot.png'):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    # LEGEND_SIZE = 10
    LEGEND_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    i = 0
    methods = filter(lambda x: (x[0] != '.' and x.find(filer_pass_word) >= 0), get_immediate_subdirectories(base))
    # methods = filter(lambda x: (x[0] != '.'), get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: str(sequence[x]) if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/rewards.npy'))
                if method.find('_BC_') >= 0:
                    rs.append(r)
                else:
                    rs.append(getRewardsSingle(r[:min(ep, len(r))], window=WINDOW))
            except Exception as e:
                continue

        if method.find('_BC_') >= 0:
            plt.plot([0, ep], [np.concatenate(rs).mean(), np.concatenate(rs).mean()],
                     color=color_map[method] if method in color_map else colors[i],
                     label=name_map[method] if method in name_map else method)
        else:
            method_shot = method[:method.find('_pybullet')] if method.find('_pybullet') != -1 else method
            plotLearningCurveAvg(rs, WINDOW, label=name_map[method_shot]
                                 if method_shot in name_map else method_shot,
                                 color=color_map[method] if method in color_map else colors[i],
                                 linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1

    # plt.plot([0, ep], [RANDOM, RANDOM], color='k', label='random')
    plt.legend(loc=4, facecolor='w', fontsize=LEGEND_SIZE, framealpha=0.6)
    plt.xlabel('number of grasps')
    plt.ylabel('success rate per ' + str(WINDOW) + ' grasps')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(base, figname), bbox_inches='tight', pad_inches=0)


def plotEvalCurve1(base, ep=50000, use_default_cm=False, filer_pass_word='_', figname='plot.png'):
    plt.style.use('ggplot')
    fig = plt.figure(dpi=300)

    gs = fig.add_gridspec(1, 5, wspace=0)
    # ax0, ax1, ax2 = gs.subplots()
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[0, 3:5])
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    LEGEND_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    i = 0
    methods = filter(lambda x: (x[0] != '.' and x.find(filer_pass_word) >= 0), get_immediate_subdirectories(base))
    # methods = filter(lambda x: (x[0] != '.'), get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: str(sequence[x]) if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/eval_rewards.npy'))
                if method.find('_BC_') >= 0:
                    rs.append(r)
                else:
                    rs.append(r[:min(ep, len(r))])
            except Exception as e:
                continue

        plot_cut = 150 * 5
        method_shot = method[:method.find('_pybullet')] if method.find('_pybullet') != -1 else method
        plotEvalCurveAvg(rs, freq=FREQ, ax=ax0, start_len=0, end_len=plot_cut,
                         label=name_map[method_shot]
                         if method_shot in name_map else method_shot,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else '-')
        plotEvalCurveAvg(rs, freq=FREQ, ax=ax2, start_len=plot_cut, end_len=ep,
                         label=name_map[method_shot]
                         if method_shot in name_map else method_shot,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1

    # ax0.plot([0, plot_cut], [RANDOM, RANDOM], color='k', label='random')
    ax1.plot([0, 1], [0.996, 0.996], color='gray', linestyle=':', linewidth=1)
    ax1.plot([0, 1], [0.8, 0], color='gray', linestyle=':', linewidth=1)
    # ax2.plot([plot_cut, ep], [RANDOM, RANDOM], color='k', label='random')
    ax2.legend(loc=0, facecolor='w', fontsize=LEGEND_SIZE, framealpha=0.6)
    # ax0.set_xlabel('number of grasps')
    # ax2.set_xlabel('number of grasps')
    # ax0.set_ylabel('validation success rate over ' + str(1000) + ' grasps')
    ax0.set_xlim(0, plot_cut)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xlim(plot_cut, ep)
    ax0.set_ylim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0.8, 1)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    # ax.set_xlabel('number of grasps')
    # Set common labels
    fig.text(0.5, 0.00, 'number of grasps', ha='center', va='center')
    fig.text(0.00, 0.5, 'validation success rate over ' + str(1000) + ' grasps',
             ha='center', va='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig(os.path.join(base, figname + ' validation1'), bbox_inches='tight', pad_inches=0)


def plotEvalCurve2(base, ep=50000, use_default_cm=False, filer_pass_word='_', figname='plot.png'):
    plt.style.use('ggplot')

    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    LEGEND_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    i = 0
    methods = filter(lambda x: (x[0] != '.' and x.find(filer_pass_word) >= 0), get_immediate_subdirectories(base))
    # methods = filter(lambda x: (x[0] != '.'), get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: str(sequence[x]) if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/eval_rewards.npy'))
                if method.find('_BC_') >= 0:
                    rs.append(r)
                else:
                    rs.append(r[:min(ep, len(r))])
            except Exception as e:
                continue

        # plot_cut = 150 * 5
        method_shot = method[:method.find('_pybullet')] if method.find('_pybullet') != -1 else method
        plotLearningCurveAvg(rs, WINDOW, label=name_map[method_shot]
        if method_shot in name_map else method_shot,
                             color=color_map[method] if method in color_map else colors[i],
                             linestyle=linestyle_map[method] if method in linestyle_map else '-',
                             show_success_rate=True)
        i += 1

    # plt.plot([0, ep], [RANDOM, RANDOM], color='k', label='random')
    plt.legend(loc=4, facecolor='w', fontsize=LEGEND_SIZE, framealpha=0.6)
    # plt.xlabel('number of grasps')
    # plt.ylabel('success rate per ' + str(WINDOW) + ' grasps')
    plt.xlabel('number of grasps')
    plt.ylabel('validation success rate over ' + str(1000) + ' grasps')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(base, figname + ' validation2'), bbox_inches='tight', pad_inches=0)


def showPerformance(base, filer_pass_word='_'):
    # methods = sorted(filter(lambda x: x[0] != '.', get_immediate_subdirectories(base)))
    methods = filter(lambda x: (x[0] != '.' and x.find(filer_pass_word) >= 0), get_immediate_subdirectories(base))
    for method in methods:
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/rewards.npy'))
                rs.append(r[-WINDOW:].mean())
            except Exception as e:
                continue
        print('{}: {:.3f}'.format(method, np.mean(rs)))


def plotTDErrors():
    plt.style.use('ggplot')
    colors = "bgrycmkw"
    method_map = {
        'ADET': 'm',
        'ADET+Q*': 'g',
        'DAGGER': 'k',
        'DQN': 'c',
        'DQN+guided': 'y',
        'DQN+Q*': 'b',
        'DQN+Q*+guided': 'r',
        "DQfD": 'chocolate',
        "DQfD+Q*": 'grey'
    }
    i = 0

    base = '/media/dian/hdd/unet/perlin'
    for method in sorted(get_immediate_subdirectories(base)):
        rs = []
        if method[0] == '.' or method == 'DAGGER' or method == 'DQN':
            continue
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/td_errors.npy'))
                rs.append(getRewardsSingle(r[:120000], window=1000))
            except Exception as e:
                continue
        if method in method_map:
            plotLearningCurveAvg(rs, 1000, label=method, color=method_map[method])
        else:
            plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        # plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        i += 1

    plt.legend(loc=0)
    plt.xlabel('number of training steps')
    plt.ylabel('TD error')
    plt.yscale('log')
    # plt.ylim((0.8, 0.93))
    plt.show()


if __name__ == '__main__':
    WINDOW = 150
    # WINDOW = 60
    FREQ = int(1500 / 10)
    RANDOM = 0.185
    # folder = 'compare with baselines'
    # folder = 'compare with all baselines'
    # folder = 'dilation aperture'
    # folder = 'ablation I'
    # folder = 'n soft FC-GQ-CNN'
    # folder = 'n soft VPG'
    # folder = 'n RAD FC-GQ-CNN'
    # folder = 'n RAD VPG'
    # folder = 'FC-GQ-CNN'
    # folder = 'VPG'
    # folder = 'n RAD rot FCN'
    # folder = 'RSS_runs'
    # folder = 'curiosity'
    # folder = 'curiosity BCE'
    folder = 'curiosity l2'
    base = '/home/zxp-s-works/grasp_data/' \
           + folder
    # base = '/home/ur5/rosws37/src/eqvar_grasp/real_world_results/' \
    #        + folder
    plotLearningCurve(base, 1500, filer_pass_word=' ', figname=folder)
    plotEvalCurve1(base, 1500, filer_pass_word=' ', figname=folder)
    plotEvalCurve2(base, 1500, filer_pass_word=' ', figname=folder)
    showPerformance(base, filer_pass_word='dqn')
