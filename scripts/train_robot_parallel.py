import concurrent.futures
import numpy.random as npr
from utils.parallel_utils import *
import os
import sys
import time
import copy
import collections
from tqdm import tqdm
from utils.visualization_utils import plot_action

sys.path.append('./')
sys.path.append('..')
from scripts.create_agent import createAgent
from utils.parameters import *
from storage.buffer import QLearningBufferExpert
import rospy
from src.envs.dual_bin_front_rear import DualBinFrontRear
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


def saveModelAndInfo(logger, agent):
    logger.saveModel(logger.num_episodes, env, agent, create_dir=True)
    logger.saveLossCurve(100)
    logger.saveRewards()
    logger.saveSGDtime()
    logger.saveLosses()
    logger.saveTdErrors()
    logger.saveStepLeftCurve(1000)
    logger.saveExpertSampleCurve(100)
    logger.saveLearningCurve(learning_curve_avg_window)
    logger.saveLearningCurve2(learning_curve_avg_window)


def train_step(agent, replay_buffer, logger):
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
        p_agent.eps = init_eps if logger.num_episodes < step_eps else final_eps
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
        # print('--------finished step', logger.num_steps, '---------')

    if is_test:
        print('test success rate: {:.2f} over {} grasps'.format(np.array(logger.rewards).mean(), max_episode))
    envs.ur5.moveToHome()
    State.set_var('main', SENTINEL)
    Action.set_var('main', SENTINEL)
    Reward.set_var('main', SENTINEL)
    Request.set_var('main', SENTINEL)
    print('training finished')


class AgentWrapper:
    """
    Wrap the agent to perform parallel thread training/ testing
    """

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
        """
        Infer action from the q maps
        Triggered by the state s
        """
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
            q_value_maps, actions_star_idx, actions_star, in_hand_obs = \
                agent.getBoltzmannActions(states, in_hands, obs, temperature=self.tau, eps=self.eps, return_patch=True)
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
        """
        Store the transition to the buffer and perform SGD step
        Triggered by reward
        """
        while True:
            reward = Reward.get_var('store_transition_SGD')
            if reward is SENTINEL or self.all_state is SENTINEL:
                break

            states, in_hands, obs = self.all_state
            buffer_obs = getCurrentObs(in_hands, obs)
            steps_lefts = envs.getStepLeft()
            dones = torch.tensor(False, dtype=torch.float32).view(1)

            for i in range(num_processes):
                is_expert = False
                data = ExpertTransition(states[i], buffer_obs[i], self.actions_star_idx[i], reward[i], states[i],
                                        buffer_obs[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
                augmentData2Buffer(replay_buffer, data, agent.rzs,
                                   onpolicy_data_aug_n, onpolicy_data_aug_rotate, onpolicy_data_aug_flip)

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
                    train_step(agent, replay_buffer, logger)
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


if __name__ == '__main__':
    format_ = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format_, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.getLogger().setLevel(logging.DEBUG)
    rospy.init_node('robot_exp')

    start_time = time.time()
    if seed is not None:
        set_seed(seed)

    # setup environment
    envs = DualBinFrontRear(ws_center=robot_ws_center, ws_x=real_workspace_size, ws_y=real_workspace_size,
                            cam_size=(cam_size, cam_size), action_sequence=action_sequence,
                            in_hand_mode='raw', pick_offset=pick_place_offset,
                            place_offset=pick_place_offset, in_hand_size=patch_size,
                            obs_source=obs_source, safe_z_region=safe_z_region, bin_size=workspace_size,
                            place_open_pos=place_open_pos, bin_size_pixel=112, z_heuristic=env_config['z_heuristic'])
    envs.State = State
    envs.Action = Action
    envs.Reward = Reward
    envs.IsRobotReady = IsRobotReady
    envs.SENTINEL = SENTINEL
    env_config['render'] = True

    # setup agent
    agent = createAgent()
    agent.train()
    if load_model_pre:
        agent.loadModel(load_model_pre)

    # setup the parallel agent
    p_agent = AgentWrapper()
    p_agent.eps = init_eps
    replay_buffer = QLearningBufferExpert(buffer_size)

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
    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), env, agent, replay_buffer)

    # starting parallel training
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(p_agent.get_action)
        executor.submit(p_agent.store_transition_SGD)
        executor.submit(main)
