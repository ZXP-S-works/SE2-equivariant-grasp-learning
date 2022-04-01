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


def addPerlinNoiseToObs(obs, c):
    for i in range(num_processes):
        obs[i, 0] += (c * rand_perlin_2d((128, 128), (
            int(np.random.choice([1, 2, 4, 8], 1)[0]),
            int(np.random.choice([1, 2, 4, 8], 1)[0]))) + c)


def addPerlinNoiseToInHand(in_hand, c):
    if in_hand_mode != 'proj':
        for i in range(num_processes):
            in_hand[i, 0] += (c * rand_perlin_2d((24, 24), (
                int(np.random.choice([1, 2], 1)[0]),
                int(np.random.choice([1, 2], 1)[0]))) + c)


def getCurrentObs(in_hand, obs):
    obss = []
    for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
    return obss


def train_step(agent, replay_buffer, logger, p_beta_schedule):
    if buffer_type == 'per' or buffer_type == 'per_expert':
        beta = p_beta_schedule.value(logger.num_steps)
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


def evaluate(envs, agent, logger):
    states, in_hands, obs = envs.reset()
    evaled = 0
    # temp_reward = [[] for _ in range(num_eval_episodes)]
    eval_rewards = []
    top5_rewards = []
    top_i = 0
    episodes = 0
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes)
    while evaled < num_eval_episodes:
        # q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, final_eps)
        q_value_maps, actions_star_idx, actions_star = agent.getBoltzmannActions(states, in_hands, obs, final_eps)
        # states_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)
        rewards = rewards.numpy()
        if eval_num_processes == 1:
            if dones.numpy() == 1:
                top_i = 0
                episodes += 1
            else:
                top_i += 1
            if top_i < 5:
                top5_rewards.append(rewards)
        if env_config['reward_type'] == 'dense':
            dones = np.ones_like(rewards)
        else:
            dones = dones.numpy()
        states = copy.copy(states_)
        obs = copy.copy(obs_)
        # for i, r in enumerate(rewards.reshape(-1)):
        #     temp_reward[i].append(r)
        evaled += int(np.sum(dones))
        # for i, d in enumerate(dones.astype(bool)):
        #     if d:
        #         R = 0
        #         for r in reversed(temp_reward[i]):
        #             R = r + gamma * R
        #         eval_rewards.append(R)
        #         temp_reward[i] = []
        eval_rewards.append(rewards)
        if not no_bar:
            eval_bar.update(evaled - eval_bar.n)
    eval_rewards = np.concatenate(eval_rewards)
    logger.eval_rewards.append(eval_rewards.mean())
    if eval_num_processes == 1:
        top5_rewards = np.concatenate(top5_rewards)
        avg_reward = eval_rewards.mean()
        std_reward = eval_rewards.std()
        avg_top5_reward = top5_rewards.mean()
        std_top5_reward = top5_rewards.std()
        print('validation: avg_reward: {:.3f} +/-{:.3f} avg_top5_reward: {:.3f} +/-{:.3f}'.
              format(avg_reward, std_reward, avg_top5_reward, std_top5_reward))

    if not no_bar:
        eval_bar.close()


def saveModelAndInfo(logger, agent):
    logger.saveModel(logger.num_steps, env, agent)
    logger.saveLossCurve(100)
    logger.saveTdErrorCurve(100)
    logger.saveRewards()
    logger.saveSGDtime()
    logger.saveLosses()
    logger.saveTdErrors()
    logger.saveStepLeftCurve(1000)
    logger.saveExpertSampleCurve(100)
    logger.saveLearningCurve(learning_curve_avg_window)
    logger.saveLearningCurve2(learning_curve_avg_window)
    logger.saveEvalCurve()
    logger.saveEvalRewards()


def train():
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
    # envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    env_config['render'] = True
    # eval_envs = EnvWrapper(eval_num_processes, simulator, env, env_config, planner_config)

    # setup agent
    agent = createAgent()
    agent.train()

    if load_model_pre:
        agent.loadModel(load_model_pre)

    # logging
    simulator_str = copy.copy(simulator)
    if simulator == 'pybullet':
        simulator_str += ('_' + robot)
    log_dir = os.path.join(log_pre, '{}_{}_{}_{}_{}'.format(alg, model, simulator_str, num_objects, max_episode_steps))
    if note:
        log_dir += '_'
        log_dir += note

    logger = Logger(log_dir, env, 'train', num_processes, max_episode, log_sub)
    hyper_parameters['model_shape'] = agent.getModelStr()
    # hyper_parameters['env_repo_hash'] = envs.getEnvGitHash()
    logger.saveParameters(hyper_parameters)
    plot_logger = logger if is_test else None

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

    envs.ur5.moveToHome()
    states, in_hands, obs = envs.reset()
    envs.p_move_reward()

    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), envs, agent, replay_buffer)

    if not no_bar:
        pbar = tqdm(total=max_episode)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    while logger.num_steps < max_episode:
        if step_eps > 0:
            if logger.num_steps < step_eps:
                eps = init_eps
            else:
                eps = final_eps
        else:
            eps = exploration.value(logger.num_steps)
        is_z_heuristic = logger.num_steps < z_heuristic_step

        if action_selection == 'egreedy':
            is_expert = 0
            q_value_maps, actions_star_idx, actions_star, in_hand_obs =\
                agent.getEGreedyActions(states, in_hands, obs, eps, return_patch=True)
        elif action_selection == 'Boltzmann':
            is_expert = 0
            q_value_maps, actions_star_idx, actions_star, in_hand_obs = \
                agent.getBoltzmannActions(states, in_hands, obs, temperature=train_tau, eps=eps,
                                          is_z_heuristic=is_z_heuristic, return_patch=True)
        else:
            raise NotImplementedError

        buffer_obs = getCurrentObs(in_hands, obs)

        primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: action_sequence.find(a),
                                                          ['p', 'x', 'y', 'z', 'r'])

        if render:
        # if render and not rewards:
            plot_action(obs, agent, actions_star, actions_star_idx, q_value_maps, num_rotations,
                        patch_size, None, in_hand_obs, action_sequence, logger=plot_logger)

        action = [*list(map(lambda x: x.item(), actions_star[0])), 0]
        states_, in_hands_, obs_, rewards, dones = envs.step(action)

        # states_, in_hands_, obs_, rewards, dones = envs.stepWait()
        steps_lefts = envs.getStepLeft()
        # steps_lefts = torch.tensor(100).view(1)

        # done_idxes = torch.nonzero(dones).squeeze(1)
        # if done_idxes.shape[0] != 0:
        #     reset_states_, reset_in_hands_, reset_obs_ = envs.reset_envs(done_idxes)
        #     for j, idx in enumerate(done_idxes):
        #         states_[idx] = reset_states_[j]
        #         in_hands_[idx] = reset_in_hands_[j]
        #         obs_[idx] = reset_obs_[j]

        buffer_obs_ = getCurrentObs(in_hands_, obs_)

        if not fixed_buffer:
            for i in range(num_processes):
                if success_to_expert:
                    is_expert = rewards[i]
                if onpolicy_data_D4_aug:
                    assert onpolicy_data_aug_n in [0, 1]
                    data = ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], states_[i],
                                            buffer_obs_[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
                    augmentData2BufferD4(replay_buffer, data, agent.rzs)
                elif onpolicy_data_aug_n > 1:
                    data = ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], states_[i],
                                            buffer_obs_[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
                    augmentData2Buffer(replay_buffer, data, agent.rzs,
                                       onpolicy_data_aug_n, onpolicy_data_aug_rotate, onpolicy_data_aug_flip)
                else:
                    replay_buffer.add(
                        ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], states_[i],
                                         buffer_obs_[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
                    )
        logger.stepBookkeeping(rewards.numpy(), steps_lefts.numpy(), dones.numpy())

        if logger.num_steps >= training_offset:
            for training_iter in range(training_iters):
                SGD_start = time.time()
                train_step(agent, replay_buffer, logger, p_beta_schedule)
                logger.SGD_time.append(time.time() - SGD_start)

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = copy.copy(in_hands_)

        if (time.time() - start_time) / 3600 > time_limit:
            break

        logger.num_steps += 1
        if not no_bar:
            timer_final = time.time()
            description = 'Steps:{}; Reward:{:.03f}; Eval Reward:{:.03f}; Explore:{:.03f}; Loss:{:.03f}; Time:{:.03f}'.format(
                logger.num_steps, logger.getCurrentAvgReward(learning_curve_avg_window),
                logger.eval_rewards[-1] if len(logger.eval_rewards) > 0 else 0,
                eps, float(logger.getCurrentLoss()), timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_steps - pbar.n)

        # if logger.num_training_steps > 0 and logger.num_steps % eval_freq == eval_freq - 1:
        #     evaluate(eval_envs, agent, logger)

        if logger.num_steps % (max_episode // num_saves) == 0:
            saveModelAndInfo(logger, agent)
            logger.saveCheckPoint(args, envs, agent, replay_buffer, save_envs=False)

    # evaluate(eval_envs, agent, logger)
    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(args, envs, agent, replay_buffer, save_envs=False)
    envs.close()


if __name__ == '__main__':
    train()
