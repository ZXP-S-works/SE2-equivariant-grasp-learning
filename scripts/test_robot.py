import os
import sys
import time
import copy
import math
import collections

import numpy as np
import rospy
import torch

import matplotlib.pyplot as plt
from src.envs.dual_bin_front_rear import DualBinFrontRear
from tqdm import tqdm

from utils.parameters import *
from scripts.create_agent import createAgent
from utils.visualization_utils import plot_action

ExpertTransition = collections.namedtuple('ExpertTransition',
                                          'state obs action reward next_state next_obs done step_left expert')

if __name__ == '__main__':
    plt.style.use('default')
    rospy.init_node('robot_exp')

    env = DualBinFrontRear(ws_center=robot_ws_center, ws_x=real_workspace_size, ws_y=real_workspace_size,
                           cam_size=(cam_size, cam_size), action_sequence=action_sequence,
                           in_hand_mode='raw', pick_offset=pick_place_offset,
                           place_offset=pick_place_offset, in_hand_size=patch_size,
                           obs_source=obs_source, safe_z_region=safe_z_region, bin_size=workspace_size,
                           place_open_pos=place_open_pos, z_heuristic=env_config['z_heuristic'])

    agent = createAgent()
    agent.train()
    agent.loadModel(load_model_pre)
    # agent.eval()

    if not no_bar:
        pbar = tqdm(total=max_episode)
        pbar.set_description('Episodes:0; Avg Reward:0.0; Reward:0.0; Time:0.0')
    timer_start = time.time()

    env.ur5.moveToHome()
    states, in_hands, obs = env.reset()

    j = 0
    pre_action = None
    reward_hist = np.asarray([])
    rewards = 1
    for step in range(max_episode):
        j += 1
        # obs, in_hand = env.getObs(pre_action)
        # addPerlinNoiseToObs(obs, 0.005)
        # addPerlinNoiseToInHand(in_hand, 0.005)
        # state = torch.tensor([env.ur5.holding_state], dtype=torch.float32)
        # q_map, action_idx, action = agent.getEGreedyActions(state, in_hand, obs, 0, 0)
        test_temp = test_tau if rewards else test_temp + train_tau
        q_value_maps, actions_star_idx, actions_star, in_hand_obs = \
            agent.getBoltzmannActions(states, in_hands, obs, test_temp, return_patch=True)

        action = [*list(map(lambda x: x.item(), actions_star[0])), 0]
        states, in_hands, obs_, rewards, dones = env.step(action)

        # if render:
        if render and not rewards:
            plot_action(obs, agent, actions_star, actions_star_idx, q_value_maps, num_rotations,
                        patch_size, None, in_hand_obs, action_sequence)

        obs = obs_
        reward_hist = np.append(reward_hist, rewards)
        pre_action = action

        if not no_bar:
            timer_final = time.time()
            description = 'Grasp{}: r_avg={:.03f}; r={:}; eps={:.03f}; t_episode{:.03f}'\
                .format(step + 1, reward_hist.mean(), rewards.item(), test_temp, timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(1)

        # if step % 10 == 9:
        #     print('test success rate: {:.2f} over {} grasps'.format(reward_hist.mean(), step + 1))

    env.close()
    print('test success rate: {:.2f} over {} grasps'.format(reward_hist.mean(), max_episode))
