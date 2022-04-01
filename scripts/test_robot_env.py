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

# if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

sys.path.append('./')
sys.path.append('..')
# from scripts.create_agent import createAgent
from utils.parameters import *
import rospy
from storage.buffer import QLearningBufferExpert, QLearningBuffer
from storage.per_buffer import PrioritizedQLearningBuffer, EXPERT, NORMAL
# from _helping_hands_rl_envs import env_factory
from src.envs.dual_bin_front_rear import DualBinFrontRear
from utils.logger import Logger
from utils.schedules import LinearSchedule
from utils.torch_utils import rand_perlin_2d
# from sl_utils.env_wrapper import EnvWrapper
from utils.torch_utils import augmentBuffer
from utils.torch_utils import augmentData2BufferD4
from utils.torch_utils import augmentData2Buffer
from networks.models import ResUCatShared, CNNShared, UCat, CNNSepEnc, CNNPatchOnly
from networks.equivariant_models_refactor import EquResUExpand, EquResUDFReg, EquResUDFRegNOut, EquShiftQ2Expand, \
    EquShiftQ2DF, EquShiftQ2Expand2, EquShiftQ2DF2, EquResUDFRegMix, EquResU, EquShiftQ2, EquShiftQ2DF2_0, \
    EquShiftQ2Expand2_0, EquShiftQ2DF3, EquResUExpand, EquResUDFReg, EquResUReg, EquResURegCas, \
    EquResUDFRegNOut, EquShiftQ2Expand, EquShiftQ2DF, EquShiftQ2Expand2, EquShiftQ2DF2, EquResUDFRegMix, EquResU, \
    EquShiftQ2, EquShiftQ2DF2_0, EquShiftQ2Expand2_0, EquShiftQ2DF3, EquShiftQ23, EquShiftQ23Cas, EquShiftQ2DF4, \
    EquShiftQ2DF5, EquResURegVers
from networks.models import ResUCatShared, CNNShared, UCat, CNNSepEnc, CNNPatchOnly, CNN
from agents.agents_4d.dqn_4d_asr import DQN4DASR
from agents.agents_4d.dqn_4d_asr_deictic import DQN4DASRDeictic
from agents.agents_3d.dqn_3d_fcn import DQN3DFCN
from agents.agents_3d.margin_3d_fcn import Margin3DFCN
from agents.agents_3d.policy_3d_fcn import Policy3DFCN
from agents.agents_3d.dqn_3d_fcn_si import DQN3DFCNSingleIn
from agents.agents_3d.policy_3d_fcn_si import Policy3DFCNSingleIn
from agents.agents_3d.margin_3d_fcn_si import Margin3DFCNSingleIn
from agents.agents_3d.dqn_3d_asr import DQN3DASR
from agents.agents_3d.margin_3d_asr import Margin3DASR
from agents.agents_3d.policy_3d_asr import Policy3DASR
from agents.agents_3d.dqn_3d_asr_deictic import DQN3DASRDeictic
from agents.agents_3d.margin_3d_asr_deictic import Margin3DASRDeictic
from agents.agents_3d.policy_3d_asr_deictic import Policy3DASRDeictic
from agents.agents_3d.dqn_3d_asr_sepenc import DQN3DASRSepEnc
from agents.agents_3d.policy_3d_asr_sepenc import Policy3DASRSepEnc
from agents.agents_3d.dqn_3d_asr_aug import DQN3DASRAug
from agents.agents_3d.margin_3d_asr_aug import Margin3DASRAug
from agents.agents_3d.dqn_3d_asr_deictic_aug import DQN3DASRDeicticAug
from agents.agents_3d.margin_3d_asr_deictic_aug import Margin3DASRDeicticAug


class RandAgent:
    def __init__(self):
        self.hm_thres = 0.015
        # self.center = [(workspace[0] + workspace[1]) / 2,
        #                (workspace[2] + workspace[3]) / 2,
        #                (workspace[4] + workspace[5]) / 2]
        # self.range = [workspace[1] - workspace[0],
        #               workspace[3] - workspace[2],
        #               workspace[5] - workspace[4]]

    def get_action(self, env):
        while True:
            _action = [np.random.uniform(workspace[0, 0], workspace[0, 1]),
                       np.random.uniform(workspace[1, 0], workspace[1, 1]),
                       np.random.randint(0, num_zs) * np.pi / num_zs, 0]  # xyrp
            if not env.isActionEmpty(_action, self.hm_thres):
                break
        return _action


if __name__ == '__main__':
    plt.style.use('default')
    rospy.init_node('robot_exp')
    global alg, model, equi_n, q2_model, load_model_pre, action_sequence, num_rotations, in_hand_mode, workspace, \
        heightmap_size, workspace_size, patch_size, q2_input, final_eps
    action_sequence = 'xyrp'
    # robot_ws_center = [-0.3426, 0.048, -0.111]  # broader not safe
    # robot_ws_center = [-0.3426, 0.048, -0.100]  # broader not safe
    robot_ws_center = [-0.3426, 0.048, -0.106]  # broader not safe
    # robot_ws_center = [-0.3426, 0.048, -0.085]  # broader safe
    pick_offset = 0.1
    place_offset = 0.25
    cam_size = 256
    real_workspace_size = 0.8
    workspace_size = 0.25
    num_rotations = 16

    real_workspace = np.asarray([[robot_ws_center[0] - real_workspace_size / 2, robot_ws_center[0] + real_workspace_size / 2],
                                 [robot_ws_center[1] - real_workspace_size / 2, robot_ws_center[1] + real_workspace_size / 2],
                                 [robot_ws_center[2], robot_ws_center[2] + 0.4]])
    workspace = np.asarray([[-workspace_size / 2, workspace_size / 2],
                            [-workspace_size / 2, workspace_size / 2],
                            [robot_ws_center[2], robot_ws_center[2] + 0.4]])
    render = True
    obs_source = 'reconstruct'
    safe_z_region = 1 / 4
    place_open_pos = 0
    pixel_size = real_workspace_size / cam_size

    env = DualBinFrontRear(ws_center=robot_ws_center, ws_x=real_workspace_size, ws_y=real_workspace_size,
                            cam_size=(cam_size, cam_size), action_sequence=action_sequence,
                            in_hand_mode='raw', pick_offset=pick_place_offset,
                            place_offset=pick_place_offset, in_hand_size=patch_size,
                            obs_source=obs_source, safe_z_region=safe_z_region,
                            place_open_pos=place_open_pos, bin_size_pixel=112)

    rospy.sleep(1)
    env.ur5.moveToHome()
    states, in_hands, obs = env.reset()

    agent = RandAgent()

    j = 0
    pre_action = None
    for step in range(max_episode):
        j += 1
        actions_star = agent.get_action(env)

        primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: action_sequence.find(a), ['p', 'x', 'y', 'z', 'r'])
        if render:
            # if render and not rewards:
            # old_data visualization
            fig = plt.figure()
            plt.imshow(obs[0, 0], cmap='gray')
            plt.colorbar()
            cos, sin = np.cos(actions_star[rot_idx]), np.sin(actions_star[rot_idx])
            x = int(actions_star[0] / pixel_size + 56)
            y = int(actions_star[1] / pixel_size + 56)
            plt.quiver(y, x, cos, sin, color='r', scale=5)
            fig.tight_layout()
            plt.show()

        action = actions_star
        states, in_hands, obs, rewards, dones = env.step(action)
        pre_action = action
        print('step: {} reward: {}'.format(step, rewards))

    env.close()
