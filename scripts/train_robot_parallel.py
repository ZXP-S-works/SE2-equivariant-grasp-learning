import concurrent.futures
import logging
import threading
import time
import numpy.random as npr
from utils.parallel_utils import *
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

from utils.visualization_utils import plot_action

sys.path.append('./')
sys.path.append('..')
from scripts.create_agent import createAgent
from utils.parameters import *
from storage.buffer import QLearningBufferExpert, QLearningBuffer
from storage.per_buffer import PrioritizedQLearningBuffer, EXPERT, NORMAL
# from _helping_hands_rl_envs import env_factory
import rospy
from src.envs.dual_bin_front_rear import DualBinFrontRear
from utils.logger import Logger
from utils.schedules import LinearSchedule
from utils.torch_utils import rand_perlin_2d
# from sl_utils.env_wrapper import EnvWrapper
from utils.torch_utils import augmentBuffer
from utils.torch_utils import augmentData2BufferD4
from utils.torch_utils import augmentData2Buffer

ExpertTransition = collections.namedtuple('ExpertTransition',
                                          'state obs action reward next_state next_obs done step_left expert')


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getCurrentObs(in_hand, obs):
    obss = []
    for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
    return obss


def saveModelAndInfo(logger, agent):
    logger.saveModel(logger.num_episodes, env, agent, create_dir=True)
    logger.saveLossCurve(100)
    # logger.saveTdErrorCurve(100)
    logger.saveRewards()
    logger.saveSGDtime()
    logger.saveLosses()
    logger.saveTdErrors()
    logger.saveStepLeftCurve(1000)
    logger.saveExpertSampleCurve(100)
    logger.saveLearningCurve(learning_curve_avg_window)
    logger.saveLearningCurve2(learning_curve_avg_window)
    # logger.saveEvalCurve()
    # logger.saveEvalRewards()


def train_step(agent, replay_buffer, logger, p_beta_schedule):
    if buffer_type == 'per' or buffer_type == 'per_expert':
        beta = p_beta_schedule.value(logger.num_episodes)
        batch, weights, batch_idxes = replay_buffer.sample(batch_size, beta,
                                                           onpolicydata=sample_onpolicydata,
                                                           onlyfailure=onlyfailure)
        loss, td_error = agent.update(batch)
        new_priorities = np.abs(td_error.cpu()) + torch.stack([t.expert for t in batch]) * per_expert_eps + per_eps
        replay_buffer.update_priorities(batch_idxes, new_priorities)
        logger.expertSampleBookkeeping(
            torch.tensor(list(zip(*batch))[-1]).sum().float().item() / batch_size)
    else:
        batch = replay_buffer.sample(batch_size, onpolicydata=sample_onpolicydata, onlyfailure=onlyfailure)
        loss, td_error = agent.update(batch)

    logger.trainingBookkeeping(loss, td_error.mean().item())
    logger.num_training_steps += 1


def main():
    if not no_bar:
        pbar = tqdm(total=max_episode)
        print('\n')
        pbar.set_description('Steps:0; Avg Reward:0.0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    envs.ur5.moveToHome()
    envs.p_reset()
    envs.p_move_reward(return_reward=False)
    rospy.sleep(1)
    while logger.num_steps < max_episode:
        if step_eps > 0:
            if logger.num_steps < step_eps:
                p_agent.eps = init_eps
            else:
                p_agent.eps = final_eps
        else:
            p_agent.eps = exploration.value(logger.num_steps)
        action = Action.get_var('main')
        IsRobotReady.get_var('main')
        # print('getting action')
        envs.p_picking(action)
        # print('picked')
        t_move_reward_start = time.time()
        reward = envs.p_move_reward()
        t_move_reward = time.time() - t_move_reward_start
        rospy.sleep(max(1 - t_move_reward, 0.1))
        # print('checked reward')
        # if logger.num_steps % 100 == 99:
        #     envs.ur5.gripper.openGripper()
        #     a = 1  # sample obj
        logger.num_steps += 1
        envs.p_place_move_center(is_request=(logger.num_steps != max_episode))
        IsSGDFinished.get_var('main')
        # print('placed')
        if not no_bar:
            timer_final = time.time()
            r_avg = np.array(logger.rewards).mean() if is_test \
                else logger.getCurrentAvgReward(learning_curve_avg_window)
            description = 'Grasp{}: r_avg={:.03f}; r={:}; eps={:.01f}; ' \
                          'loss={:.03f}; t_episode={:.02f}'.format(
                logger.num_steps, r_avg,
                reward, p_agent.eps, float(logger.getCurrentLoss()), timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_steps - pbar.n)
        if logger.num_steps % (max_episode // num_saves) == 0 or logger.num_steps == max_episode:
            saveModelAndInfo(logger, agent)
            logger.saveCheckPoint(args, envs, agent, replay_buffer, save_envs=False)
        # print('SGD finished')
        # 4print('--------finished step', logger.num_steps, '---------')

    if is_test:
        print('test success rate: {:.2f} over {} grasps'.format(np.array(logger.rewards).mean(), max_episode))
    envs.ur5.moveToHome()
    # saveModelAndInfo(logger, agent)
    State.set_var('main', SENTINEL)
    Action.set_var('main', SENTINEL)
    Reward.set_var('main', SENTINEL)
    Request.set_var('main', SENTINEL)
    print('training finished')


class AgentWrapper:
    def __init__(self):
        self.all_state = None  # all_state include (grasping_state, in_hands, obs)
        self.network = threading.Lock()
        self.obs = None
        self.in_hand_obs = None
        self.actions_star = None
        self.q_value_maps = None
        self.actions_star_idx = None
        self.eps = 1.
        self.last_grasp_succeed = -1
        if is_test:
            self.plot_logger = logger
            self.tau = test_tau
        else:
            self.plot_logger = None
            self.tau = train_tau
        primative_idx, x_idx, y_idx, z_idx, self.rot_idx = map(lambda a: action_sequence.find(a),
                                                               ['p', 'x', 'y', 'z', 'r'])

    def get_action(self):
        while True:
            state = State.get_var('get_action')
            # print('got state')
            if state is SENTINEL:
                break
            # print('get_action getting network')
            self.network.acquire()
            # print('get_action got network')
            self.all_state = state
            states, in_hands, obs = self.all_state
            in_hand_obs = None
            if action_selection == 'egreedy':
                q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, self.eps)
            elif action_selection == 'Boltzmann':  # in this case eps is temperature, interpolates from large to small
                q_value_maps, actions_star_idx, actions_star, in_hand_obs = \
                    agent.getBoltzmannActions(states, in_hands, obs, temperature=self.tau, eps=self.eps,
                                              is_z_heuristic=is_z_heuristic, return_patch=True)
            else:
                raise NotImplementedError
            self.network.release()
            # print('got action')
            action = [*list(map(lambda x: x.item(), actions_star[0])), 0]
            # print('agent s action', actions_star_idx)
            self.obs = obs
            self.in_hand_obs = in_hand_obs
            self.actions_star = actions_star
            self.actions_star_idx = actions_star_idx
            self.q_value_maps = q_value_maps
            Action.set_var('get_action', action)
        print('get_action killed')

    def store_transition_SGD(self):
        while True:
            reward = Reward.get_var('store_transition_SGD')
            if reward is SENTINEL or self.all_state is SENTINEL:
                break

            states, in_hands, obs = self.all_state
            buffer_obs = getCurrentObs(in_hands, obs)
            steps_lefts = envs.getStepLeft()
            dones = torch.tensor(False, dtype=torch.float32).view(1)

            for i in range(num_processes):
                if success_to_expert:
                    is_expert = reward[i]
                else:
                    is_expert = False
                if onpolicy_data_D4_aug:
                    assert onpolicy_data_aug_n in [0, 1]
                    data = ExpertTransition(states[i], buffer_obs[i], self.actions_star_idx[i], reward[i], states[i],
                                            buffer_obs[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
                    augmentData2BufferD4(replay_buffer, data, agent.rzs)
                elif onpolicy_data_aug_n > 1:
                    data = ExpertTransition(states[i], buffer_obs[i], self.actions_star_idx[i], reward[i], states[i],
                                            buffer_obs[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
                    augmentData2Buffer(replay_buffer, data, agent.rzs,
                                       onpolicy_data_aug_n, onpolicy_data_aug_rotate, onpolicy_data_aug_flip)
                else:
                    replay_buffer.add(
                        ExpertTransition(states[i], buffer_obs[i], self.actions_star_idx[i], reward[i], states[i],
                                         buffer_obs[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
                    )
            logger.stepBookkeeping(reward.numpy(), steps_lefts.numpy(), dones.numpy())
            # print('transition augmented')
            # if logger.num_episodes >= training_offset:
            # print('store transition SGD getting network')
            self.network.acquire()
            # print('store transition SGD got network')
            if (not is_test and logger.num_episodes >= training_offset) \
                    or (is_test and is_baseline and not reward) \
                    or (is_test and not is_baseline):
                for training_iter in range(training_iters):
                    SGD_start = time.time()
                    train_step(agent, replay_buffer, logger, p_beta_schedule)
                    logger.SGD_time.append(time.time() - SGD_start)
            if is_baseline and (self.last_grasp_succeed == 0 and reward == 1):
                agent.loadModel(load_model_pre)
            self.network.release()
            # print('trained a SGD step')
            # print('training')
            self.last_grasp_succeed = reward
            IsSGDFinished.set_var('store_transition_SGD', True)
            # if render and not reward:
            if render:
                plot_action(obs, agent, self.actions_star, self.actions_star_idx, self.q_value_maps, num_rotations,
                            patch_size, None, self.in_hand_obs, action_sequence, logger=self.plot_logger)
        print('store_transition_SGD killed')

    # def render(self):
    #     plot_action(obs, agent, actions_star, actions_star_idx, q_value_maps, num_rotations,
    #                 patch_size, rewards, in_hand_obs, action_sequence)
    #     fig, axs = plt.subplots(figsize=(8, 6), nrows=2, ncols=2)
    #     obs1 = axs[0][0].imshow(self.obs[0][0, 0][agent.inward_padding:-agent.inward_padding,
    #                             agent.inward_padding:-agent.inward_padding], cmap='gray')
    #     axs[0][0].scatter(self.actions_star_idx[0, 1], self.actions_star_idx[0, 0], c='r', marker='*')
    #     axs[0][0].axis('off')
    #     q1 = axs[0][1].imshow(self.q_value_maps[0][0])
    #     axs[0][1].axis('off')
    #     obs2 = axs[1][0].imshow(self.obs[1][0, 0], cmap='gist_gray')
    #     theta = self.actions_star_idx[0, self.rot_idx] * np.pi / num_rotations
    #     cos, sin = np.cos(theta), np.sin(theta)
    #     axs[1][0].quiver(patch_size / 2, patch_size / 2, cos, sin, color='r', scale=5)
    #     axs[1][0].axis('off')
    #     q2 = axs[1][1].imshow(self.q_value_maps[1])
    #     axs[1][1].axis('off')
    #     fig.colorbar(obs1, ax=axs[0][0])
    #     fig.colorbar(q1, ax=axs[0][1])
    #     fig.colorbar(obs2, ax=axs[1][0])
    #     fig.colorbar(q2, ax=axs[1][1])
    #     axs[0][0].title.set_text('Q1 Observation')
    #     axs[0][1].title.set_text('Q1')
    #     axs[1][0].title.set_text('Q2 Observation')
    #     axs[1][1].title.set_text('Q2')
    #     # is_grasp_success = 'Grasp succeed' if rewards.sum() > 0 else 'Grasp failed'
    #     # fig.suptitle(is_grasp_success)
    #     fig.tight_layout()
    #     plt.show()


if __name__ == '__main__':
    format_ = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format_, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.getLogger().setLevel(logging.DEBUG)
    rospy.init_node('robot_exp')

    start_time = time.time()
    if seed is not None:
        set_seed(seed)

    envs = DualBinFrontRear(ws_center=robot_ws_center, ws_x=real_workspace_size, ws_y=real_workspace_size,
                            cam_size=(cam_size, cam_size), action_sequence=action_sequence,
                            in_hand_mode='raw', pick_offset=pick_place_offset,
                            place_offset=pick_place_offset, in_hand_size=patch_size,
                            obs_source=obs_source, safe_z_region=safe_z_region, bin_size=workspace_size,
                            place_open_pos=place_open_pos, bin_size_pixel=112, z_heuristic=env_config['z_heuristic'])
    envs.State = State
    envs.Action = Action
    envs.Reward = Reward
    # envs.Request = Request
    envs.IsRobotReady = IsRobotReady
    envs.SENTINEL = SENTINEL
    env_config['render'] = True

    # setup agent
    agent = createAgent()
    agent.train()
    if load_model_pre:
        agent.loadModel(load_model_pre)

    # eps = init_eps

    # logging
    simulator_str = copy.copy(simulator)
    if simulator == 'pybullet':
        simulator_str += ('_' + robot)
    log_dir = os.path.join(log_pre, '{}_{}_{}_{}_{}'.format(alg, model, simulator_str, num_objects, max_episode_steps))
    if note:
        log_dir += '_'
        log_dir += note
    logger = Logger(log_dir, env, 'train', num_processes, max_episode, log_sub)
    logger.saveParameters(hyper_parameters)

    p_agent = AgentWrapper()
    p_agent.eps = init_eps

    # is_z_heuristic = True
    is_z_heuristic = logger.num_episodes < z_heuristic_step

    if buffer_type == 'per':
        replay_buffer = PrioritizedQLearningBuffer(buffer_size, per_alpha, NORMAL)
    elif buffer_type == 'per_expert':
        replay_buffer = PrioritizedQLearningBuffer(buffer_size, per_alpha, EXPERT)
    elif buffer_type == 'expert':
        replay_buffer = QLearningBufferExpert(buffer_size)
    else:
        replay_buffer = QLearningBuffer(buffer_size)
    exploration = LinearSchedule(schedule_timesteps=explore, initial_p=init_eps, final_p=final_eps)
    p_beta_schedule = LinearSchedule(schedule_timesteps=max_episode, initial_p=per_beta, final_p=1.0)

    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), env, agent, replay_buffer)
    # State = Pipe('state')
    # Action = Pipe('action')
    # Reward = Pipe('reward')
    # Request = Pipe('request')
    # IsSGDFinished = Pipe('is_SGD_finished')

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # executor.submit(envs.p_sensor_processing)
        executor.submit(p_agent.get_action)
        executor.submit(p_agent.store_transition_SGD)
        executor.submit(main)
