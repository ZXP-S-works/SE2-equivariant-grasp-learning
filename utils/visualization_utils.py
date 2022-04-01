import os

import matplotlib.pyplot as plt
import numpy as np


def plot_action(obs, agent, actions_star, actions_star_idx, q_value_maps, num_rotations,
                patch_size, rewards, in_hand_obs, action_sequence, is_title_success=False, logger=None):
    primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: action_sequence.find(a),
                                                      ['p', 'x', 'y', 'z', 'r'])
    if not isinstance(q_value_maps, tuple):
        fig, axs = plt.subplots(figsize=(8, 4), nrows=1, ncols=2)
        obs1 = axs[0].imshow(obs[0, 0][agent.inward_padding:-agent.inward_padding,
                             agent.inward_padding:-agent.inward_padding], cmap='gray')
        axs[0].scatter(actions_star_idx[0, 1], actions_star_idx[0, 0], c='r', marker='*')
        axs[0].axis('off')
        q1 = axs[1].imshow(q_value_maps[0][actions_star_idx[0, rot_idx]])
        axs[1].axis('off')
        fig.colorbar(obs1, ax=axs[0])
        fig.colorbar(q1, ax=axs[1])
        axs[0].title.set_text('Q Observation')
        axs[1].title.set_text('Q when a_theta=' + str(actions_star_idx[0, rot_idx].item()))
        if is_title_success:
            is_grasp_success = 'Grasp succeed' if rewards.sum() > 0 else 'Grasp failed'
            fig.suptitle(is_grasp_success)
        fig.tight_layout()
        plt.show()
    elif z_idx == -1:
        obs = (obs, in_hand_obs.to('cpu').numpy())
        fig, axs = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
        obs1 = axs[0][0].imshow(obs[0][0, 0][agent.inward_padding:-agent.inward_padding,
                                agent.inward_padding:-agent.inward_padding], cmap='gray')
        axs[0][0].scatter(actions_star_idx[0, 1], actions_star_idx[0, 0], c='r', marker='*')
        axs[0][0].axis('off')
        q1 = axs[0][1].imshow(q_value_maps[0][0])
        axs[0][1].axis('off')
        obs2 = axs[1][0].imshow(obs[1][0, 0], cmap='gist_gray')
        theta = actions_star_idx[0, rot_idx] * np.pi / num_rotations
        cos, sin = np.cos(theta), np.sin(theta)
        axs[1][0].quiver(patch_size / 2, patch_size / 2, cos, sin, color='r', scale=5)
        axs[1][0].axis('off')
        q2 = axs[1][1].imshow(q_value_maps[1])
        axs[1][1].axis('off')
        fig.colorbar(obs1, ax=axs[0][0])
        fig.colorbar(q1, ax=axs[0][1])
        fig.colorbar(obs2, ax=axs[1][0])
        fig.colorbar(q2, ax=axs[1][1])
        axs[0][0].title.set_text('Q1 Observation')
        axs[0][1].title.set_text('Q1')
        axs[1][0].title.set_text('Q2 Observation')
        axs[1][1].title.set_text('Q2')
        if is_title_success:
            is_grasp_success = 'Grasp succeed' if rewards.sum() > 0 else 'Grasp failed'
            fig.suptitle(is_grasp_success)
        fig.tight_layout()
        plt.show()
    else:
        in_hand_obs, patch_q3 = in_hand_obs
        obs = (obs, in_hand_obs.to('cpu').numpy(), patch_q3.to('cpu').numpy())
        fig, axs = plt.subplots(figsize=(8, 12), nrows=3, ncols=2)
        obs1 = axs[0][0].imshow(obs[0][0, 0][agent.inward_padding:-agent.inward_padding,
                                agent.inward_padding:-agent.inward_padding], cmap='gray')
        axs[0][0].scatter(actions_star_idx[0, 1], actions_star_idx[0, 0], c='r', marker='*')
        axs[0][0].axis('off')
        q1 = axs[0][1].imshow(q_value_maps[0][0])
        axs[0][1].axis('off')
        obs2 = axs[1][0].imshow(obs[1][0, 0], cmap='gist_gray')
        theta = actions_star_idx[0, rot_idx] * np.pi / num_rotations
        cos, sin = np.cos(theta), np.sin(theta)
        axs[1][0].quiver(patch_size / 2, patch_size / 2, cos, sin, color='r', scale=5)
        axs[1][0].axis('off')
        q2 = axs[1][1].imshow(q_value_maps[1])
        # axs[1][1](projection='polar')
        # q2 = axs[1][1].pcolormesh()
        axs[1][1].axis('off')
        obs3 = axs[2][0].imshow(obs[2][0, 0], cmap='gist_gray')
        axs[2][0].axis('off')
        q3 = axs[2][1].imshow(q_value_maps[2].detach().reshape(-1, 1).flip(0))
        z = actions_star[0, z_idx].item()
        equal_h = obs[2][0, 0] < z
        # equal_h = (z - 0.01 < obs[2][0, 0]) & (obs[2][0, 0] < z + 0.01)
        equal_h = equal_h.nonzero()
        # axs[2][1].scatter(actions_star_idx[0, z_idx], 0, c='r', marker='*')
        axs[2][0].scatter(equal_h[1], equal_h[0], c='r', alpha=0.9, s=2)
        axs[2][1].axis('off')
        fig.colorbar(obs1, ax=axs[0][0])
        fig.colorbar(q1, ax=axs[0][1])
        fig.colorbar(obs2, ax=axs[1][0])
        fig.colorbar(q2, ax=axs[1][1])
        fig.colorbar(obs3, ax=axs[2][0])
        fig.colorbar(q3, ax=axs[2][1], pad=0.35)
        axs[0][0].title.set_text('Q1 Observation')
        axs[0][1].title.set_text('Q1')
        axs[1][0].title.set_text('Q2 Observation')
        axs[1][1].title.set_text('Q2')
        axs[2][0].title.set_text('Q3 Observation')
        axs[2][1].title.set_text('Q3')
        if is_title_success:
            is_grasp_success = 'Grasp succeed' if rewards.sum() > 0 else 'Grasp failed'
            fig.suptitle(is_grasp_success)
        fig.tight_layout()
        plt.show()

    if logger is not None:
        fig.savefig(os.path.join(logger.info_dir, 'step' + str(logger.num_episodes) + '.pdf'))
