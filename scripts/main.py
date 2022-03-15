import os
import sys
import time
import copy
import collections
from tqdm import tqdm

sys.path.append('./')
sys.path.append('..')
sys.path.append('PATH_TO_Help_Hans_rl_envs')
# example PATH_TO_Help_Hans_rl_envs: sys.path.append('/home/my computer/helping_hands_rl_envs')

from scripts.create_agent import createAgent
from utils.visualization_utils import plot_action
from utils.parameters import *
from storage.buffer import QLearningBufferExpert
from utils.logger import Logger
from utils.env_wrapper import EnvWrapper
from utils.torch_utils import augmentData2Buffer
np.seterr(invalid='ignore')

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
    """
    Training an SGD step
    """
    batch = replay_buffer.sample(sample_batch_size, onpolicydata=sample_onpolicydata, onlyfailure=onlyfailure)
    loss, td_error = agent.update(batch)
    logger.trainingBookkeeping(loss, td_error.mean().item())
    logger.num_training_steps += 1


def evaluate(envs, agent, logger):
    """
    Evaluate the agent with num_eval_episodes
    """
    states, in_hands, obs = envs.reset()
    eval_steps = 0
    eval_rewards = []
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes)

    agent.eval()
    with torch.no_grad():
        while eval_steps < num_eval_episodes:
            # Boltzmann sampling an action with evaluation temperature test_tau
            q_value_maps, actions_star_idx, actions_star =\
                    agent.getBoltzmannActions(states, in_hands, obs, temperature=test_tau)
            actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
            states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)
            rewards = rewards.numpy()
            dones = np.ones_like(rewards)
            states = copy.copy(states_)
            obs = copy.copy(obs_)
            eval_steps += int(np.sum(dones))
            eval_rewards.append(rewards)
            eval_bar.update(eval_steps - eval_bar.n)
    agent.train()
    eval_rewards = np.concatenate(eval_rewards)
    logger.eval_rewards.append(eval_rewards.mean())

    if not no_bar:
        eval_bar.close()


def saveModelAndInfo(logger, agent):
    """
    Saving the model parameters and the training information
    :param logger:
    :param agent:
    """
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
    logger.saveEvalCurve()
    logger.saveEvalRewards()


def train():
    if seed is not None:
        set_seed(seed)

    # setup the environment
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    env_config['render'] = False
    eval_envs = EnvWrapper(eval_num_processes, simulator, env, env_config, planner_config)

    # setup the agent
    agent = createAgent()
    agent.train()
    if load_model_pre:
        agent.loadModel(load_model_pre)

    # setup logging
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

    # setup buffer
    replay_buffer = QLearningBufferExpert(buffer_size)

    states, in_hands, obs = envs.reset()

    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), envs, agent, replay_buffer)

    if not no_bar:
        pbar = tqdm(total=max_episode)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    # the training loop
    while logger.num_episodes < max_episode:
        eps = init_eps if logger.num_episodes < step_eps else final_eps
        is_expert = 0
        q_value_maps, actions_star_idx, actions_star, in_hand_obs = \
            agent.getBoltzmannActions(states, in_hands, obs, temperature=train_tau, eps=eps, return_patch=True)

        buffer_obs = getCurrentObs(in_hands, obs)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        envs.stepAsync(actions_star, auto_reset=False)

        states_, in_hands_, obs_, rewards, dones = envs.stepWait()
        steps_lefts = envs.getStepLeft()

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_in_hands_, reset_obs_ = envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                states_[idx] = reset_states_[j]
                in_hands_[idx] = reset_in_hands_[j]
                obs_[idx] = reset_obs_[j]

        # if render:
        if render and not rewards.item():
            plot_action(obs, agent, actions_star, actions_star_idx, q_value_maps, num_rotations,
                        patch_size, rewards, in_hand_obs, action_sequence)

        buffer_obs_ = getCurrentObs(in_hands_, obs_)

        if not fixed_buffer:
            for i in range(num_processes):
                data = ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], states_[i],
                                            buffer_obs_[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
                augmentData2Buffer(replay_buffer, data, agent.rzs,
                                   onpolicy_data_aug_n, onpolicy_data_aug_rotate, onpolicy_data_aug_flip)

        logger.stepBookkeeping(rewards.numpy(), steps_lefts.numpy(), dones.numpy())

        if logger.num_episodes >= training_offset:
            for training_iter in range(training_iters):
                SGD_start = time.time()
                train_step(agent, replay_buffer, logger)
                logger.SGD_time.append(time.time() - SGD_start)

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = copy.copy(in_hands_)

        if not no_bar:
            timer_final = time.time()
            description = 'Steps:{}; Reward:{:.03f}; Eval Reward:{:.03f}; Explore:{:.03f}; Loss:{:.03f}; Time:{:.03f}'.format(
                logger.num_steps, logger.getCurrentAvgReward(learning_curve_avg_window),
                logger.eval_rewards[-1] if len(logger.eval_rewards) > 0 else 0,
                eps, float(logger.getCurrentLoss()), timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_episodes - pbar.n)
        logger.num_steps += num_processes

        if logger.num_training_steps > 0 and logger.num_episodes % eval_freq == eval_freq - 1:
            evaluate(eval_envs, agent, logger)

        if (logger.num_episodes + 1) % (max_episode // num_saves) == 0:
            saveModelAndInfo(logger, agent)

    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(args, envs, agent, replay_buffer)
    envs.close()


if __name__ == '__main__':
    train()
