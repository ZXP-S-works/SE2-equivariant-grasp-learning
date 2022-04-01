import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm

import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from scripts.create_agent import createAgent
from scripts.main import addPerlinNoiseToInHand, addPerlinNoiseToObs
from utils.parameters import *
from storage.buffer import QLearningBufferExpert, QLearningBuffer
from helping_hands_rl_envs import env_factory
from utils.logger import Logger
from utils.schedules import LinearSchedule
from utils.torch_utils import rand_perlin_2d
from utils.env_wrapper import EnvWrapper

ExpertTransition = collections.namedtuple('ExpertTransition',
                                          'state obs action reward next_state next_obs done step_left expert')


def test():
    plt.style.use('default')
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    agent = createAgent()
    agent.train()
    agent.loadModel(load_model_pre)
    agent.eval()
    states, in_hands, obs = envs.reset()
    test_episode = 1000
    total = 0
    s = 0
    steps = 0
    step_times = []
    pbar = tqdm(total=test_episode)
    while total < 1000:
        if perlin:
            addPerlinNoiseToObs(obs, perlin)
            addPerlinNoiseToInHand(in_hands, perlin)
        if action_selection == 'egreedy':
            q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, final_eps,
                                                                                   hm_threshold=hm_threshold)
        elif action_selection == 'Boltzmann':  # final_eps here is temperature, interpolates from large to small
            q_value_maps, actions_star_idx, actions_star = agent.getBoltzmannActions(states, in_hands, obs, final_eps)
        else:
            raise NotImplementedError
        # q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, 0, 0)

        if render:
            # old_data visualization
            fig, axs = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)
            obs = axs[0].imshow(obs[0, 0])
            axs[0].axis('off')
            q1 = axs[1].imshow(q_value_maps[0])
            axs[1].axis('off')
            fig.colorbar(obs, ax=axs[0])
            fig.colorbar(q1, ax=axs[1])
            axs[0].title.set_text('Observation')
            axs[1].title.set_text('Q1')
            fig.tight_layout()
            # plt.figure()
            # plt.imshow(obs[0, 0])
            # plt.colorbar()
            # plt.figure()
            # plt.imshow(q1_value_maps[0])
            # plt.colorbar()
            # fig.tight_layout()
            plt.show()

        # plt.imshow(obs[0, 0])
        # plt.show()
        # plotQMaps(q_value_maps)
        # plotSoftmax(q_value_maps)
        # plotQMaps(q_value_maps, save='/media/dian/hdd/analysis/qmap/house1_dqfdall', j=j)
        # plotSoftmax(q_value_maps, save='/media/dian/hdd/analysis/qmap/house1_dqfd_400k', j=j)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)

        # plan_actions = envs.getPlan()
        # planner_actions_star_idx, planner_actions_star = agent.getActionFromPlan(plan_actions)
        # ranks.extend(rankOfAction(q_value_maps, planner_actions_star_idx))
        # print('avg rank of ae: {}'.format(np.mean(ranks)))

        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = in_hands_

        s += rewards.sum().int().item()

        if dones.sum():
            total += dones.sum().int().item()

        steps += num_processes
        if env_config['reward_type'] == 'dense':
            total_tries = steps
        else:
            total_tries = total

        pbar.set_description(
            '{}/{}, SR: {:.3f}'
                .format(s, total_tries, float(s) / total_tries if total_tries != 0 else 0)
        )
        pbar.update(dones.sum().int().item())

    # np.save('ranks_dqfd_all.npy', ranks)
    # plotRanks(ranks, 1200)


if __name__ == '__main__':
    test()
