import copy
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
from scripts.create_agent import createAgent
from utils.parameters import *
from utils.env_wrapper import EnvWrapper
from utils.visualization_utils import plot_action

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
    total = 0
    s = 0
    sr = torch.tensor([])
    steps = 0
    action_hist = []
    reached_button_hist = []
    pbar = tqdm(total=max_episode)
    while total < max_episode:
        if action_selection == 'egreedy':
            is_expert = 0
            q_value_maps, actions_star_idx, actions_star, in_hand_obs = \
                agent.getEGreedyActions(states, in_hands, obs, 0, return_patch=True)
        elif action_selection == 'Boltzmann':
            is_expert = 0
            q_value_maps, actions_star_idx, actions_star, in_hand_obs = \
                agent.getBoltzmannActions(states, in_hands, obs, temperature=test_tau, eps=0,
                                          is_z_heuristic=is_z_heuristic, return_patch=True)
        elif action_selection == 'asr_Boltzmann':
            is_expert = 0
            q_value_maps, actions_star_idx, actions_star, in_hand_obs = \
                agent.getBoltzmannActions(states, in_hands, obs, temperature=test_tau, eps=0,
                                          is_z_heuristic=is_z_heuristic, return_patch=True,
                                          asr_Boltzmann='asr_Boltzmann')
        elif action_selection == 'full_asr_Boltzmann':
            is_expert = 0
            q_value_maps, actions_star_idx, actions_star, in_hand_obs = \
                agent.getBoltzmannActions(states, in_hands, obs, temperature=test_tau, eps=0,
                                          is_z_heuristic=is_z_heuristic, return_patch=True,
                                          asr_Boltzmann='full_asr_Boltzmann')

        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        action_hist.append(actions_star_idx)
        # reached_button_hist.append(reached_button)

        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)

        if env_config['reward_type'] == 'dense':
            dones = np.ones_like(rewards)
        else:
            dones = dones.numpy()

        # if render:
        if render and not rewards.item():
            plot_action(obs, agent, actions_star, actions_star_idx, q_value_maps, num_rotations,
                        patch_size, rewards, in_hand_obs, action_sequence)

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = in_hands_

        s += rewards.sum().int().item()
        sr = torch.cat((sr, rewards))

        if dones.sum():
            # total += dones.sum().int().item()
            total += dones.sum()

        steps += num_processes
        if env_config['reward_type'] == 'dense':
            total_tries = steps
        else:
            total_tries = total

        pbar.set_description(
            '{}/{}, SR: {:.3f}'
                .format(s, total_tries, float(s) / total_tries if total_tries != 0 else 0)
        )
        pbar.update(dones.sum())
        # if total % 10 == 9:
        #     with open('sr_for_157_obj.json', 'w') as f:
        #         json.dump(sr, f, indent=4)

    a = 1
    # SR = np.array(sr)
    # SR = SR.reshape(-1, 100).mean(-1)
    # plt.figure()
    # plt.plot(SR)
    # sorted_sr = np.sort(SR.copy())
    # plt.figure()
    # plt.plot(sorted_sr)
    # plt.show()
    # with open('SR_for_157_obj.json', 'w') as f:
    #     json.dump(SR.tolist(), f, indent=4)
    # np.save('ranks_dqfd_all.npy', ranks)
    # plotRanks(ranks, 1200)


if __name__ == '__main__':
    test()
