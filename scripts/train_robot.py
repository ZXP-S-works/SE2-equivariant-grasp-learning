import os
import sys
import time
import copy
import collections
from tqdm import tqdm
# if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from utils.visualization_utils import plot_action
sys.path.append('./')
sys.path.append('..')
from scripts.create_agent import createAgent
from utils.parameters import *
from storage.buffer import QLearningBufferExpert, QLearningBuffer
import rospy
from utils.logger import Logger
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


def train_step(agent, replay_buffer, logger):
    batch = replay_buffer.sample(batch_size, onpolicydata=sample_onpolicydata, onlyfailure=onlyfailure)
    loss, td_error = agent.update(batch)
    logger.trainingBookkeeping(loss, td_error.mean().item())
    logger.num_training_steps += 1


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
    env_config['render'] = True

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
    logger.saveParameters(hyper_parameters)
    plot_logger = logger if is_test else None

    replay_buffer = QLearningBufferExpert(buffer_size)

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
        eps = init_eps if logger.num_episodes < step_eps else final_eps

        is_expert = 0
        q_value_maps, actions_star_idx, actions_star, in_hand_obs = \
            agent.getBoltzmannActions(states, in_hands, obs, temperature=train_tau, eps=eps, return_patch=True)

        buffer_obs = getCurrentObs(in_hands, obs)

        if render:
        # if render and not rewards:
            plot_action(obs, agent, actions_star, actions_star_idx, q_value_maps, num_rotations,
                        patch_size, None, in_hand_obs, action_sequence, logger=plot_logger)

        action = [*list(map(lambda x: x.item(), actions_star[0])), 0]
        states_, in_hands_, obs_, rewards, dones = envs.step(action)
        steps_lefts = envs.getStepLeft()
        buffer_obs_ = getCurrentObs(in_hands_, obs_)

        if not fixed_buffer:
            for i in range(num_processes):
                data = ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], states_[i],
                                            buffer_obs_[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
                augmentData2Buffer(replay_buffer, data, agent.rzs,
                                   onpolicy_data_aug_n, onpolicy_data_aug_rotate, onpolicy_data_aug_flip)
        logger.stepBookkeeping(rewards.numpy(), steps_lefts.numpy(), dones.numpy())

        if logger.num_steps >= training_offset:
            for training_iter in range(training_iters):
                SGD_start = time.time()
                train_step(agent, replay_buffer, logger)
                logger.SGD_time.append(time.time() - SGD_start)

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = copy.copy(in_hands_)

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

        if logger.num_steps % (max_episode // num_saves) == 0:
            saveModelAndInfo(logger, agent)
            logger.saveCheckPoint(args, envs, agent, replay_buffer, save_envs=False)

    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(args, envs, agent, replay_buffer, save_envs=False)
    envs.close()


if __name__ == '__main__':
    train()
