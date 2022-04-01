import numpy as np
import torch
import torch.nn.functional as F
from agents.agents_3d.dqn_3d_asr import DQN3DASR
from utils import torch_utils
from utils.parameters import is_bandit, q1_failure_td_target, td_err_measurement, q2_train_q1
from utils.torch_utils import getDrQAugmentedTransition


class DQN3DASRDrQ(DQN3DASR):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8), aug_type='cn'):
        DQN3DASR.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz,
                          rz_range)
        self.K = 2
        self.M = 2
        self.aug_type = 'se2'
        # self.aug_type = 'cn'

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pixel = action_idx[:, 0:2]
        a2_idx = action_idx[:, 2:3]
        if self.sl:
            q_targets = self.gamma ** step_lefts
        elif is_bandit:
            q_targets = rewards
        else:
            with torch.no_grad():
                q_targets = []
                for _ in range(self.K):
                    aug_next_obss = []
                    for i in range(batch_size):
                        aug_next_obs, _ = getDrQAugmentedTransition(next_obs[0][i, 0].cpu(), action_idx=None,
                                                                    rzs=self.rzs, aug_type=self.aug_type)
                        aug_next_obss.append(torch.tensor(aug_next_obs.reshape(1, 1, *aug_next_obs.shape)))
                    aug_next_obss = torch.cat(aug_next_obss, dim=0).to(self.device)
                    q1_map_prime, obs_prime_encoding = self.forwardFCN(next_states, next_obs[1], aug_next_obss,
                                                                       target_net=True)
                    x_star = torch_utils.argmax2d(q1_map_prime)
                    q2_prime = self.forwardQ2(next_states, next_obs[1], aug_next_obss, obs_prime_encoding, x_star,
                                              target_net=True)
                    q_prime = q2_prime.max(1)[0]
                    q_target = rewards + self.gamma * q_prime * non_final_masks
                    q_targets.append(q_target)
                q_targets = torch.stack(q_targets).mean(dim=0)

        self.loss_calc_dict['q_target'] = q_targets

        q1_outputs = []
        q1_preds = []
        q2_outputs = []
        q2_preds = []
        actions = []
        for _ in range(self.M):
            aug_obss = []
            aug_actions = []
            for i in range(batch_size):
                aug_obs, aug_action = getDrQAugmentedTransition(obs[0][i, 0].cpu(), action_idx[i].cpu().numpy(),
                                                                rzs=self.rzs, aug_type=self.aug_type)
                aug_obss.append(torch.tensor(aug_obs.reshape(1, 1, *aug_obs.shape)))
                aug_actions.append(aug_action)
            aug_obss = torch.cat(aug_obss, dim=0).to(self.device)
            aug_actions = torch.tensor(aug_actions).to(self.device)
            q1_output, obs_encoding = self.forwardFCN(states, obs[1], aug_obss)
            q1_pred = q1_output[torch.arange(0, batch_size), aug_actions[:, 0], aug_actions[:, 1]]
            q2_output = self.forwardQ2(states, obs[1], aug_obss, obs_encoding, aug_actions[:, :2])
            q2_pred = q2_output[torch.arange(batch_size), aug_actions[:, 2]]

            q1_outputs.append(q1_output)
            q1_preds.append(q1_pred)
            q2_outputs.append(q2_output)
            q2_preds.append(q2_pred)
            actions.append(aug_actions)
        q1_outputs = torch.cat(q1_outputs)
        q1_preds = torch.cat(q1_preds)
        q2_outputs = torch.cat(q2_outputs)
        q2_preds = torch.cat(q2_preds)
        actions = torch.cat(actions)
        self.loss_calc_dict['q1_output'] = q1_outputs
        self.loss_calc_dict['q2_output'] = q2_outputs
        self.loss_calc_dict['action'] = actions

        success_mask = rewards.long() > 0
        failure_mask = rewards.clone() < 1
        q1_success_target = q_targets[success_mask]
        # the target for q1, when grasp failed
        if q1_failure_td_target == 'rewards' or q1_failure_td_target == 'smooth_l1':
            q1_failure_target = q_targets[failure_mask]
        # else:  # second_max_q2 !!! second max q2 maybe inaccurate, should use max(q2^q2[a]) instead
        elif q1_failure_td_target in ['non_action_max_q2', 'half_non_action_max_q2']:
            # non_action_max_q2: max(q2^q2[a])
            if failure_mask.sum() != 0:
                with torch.no_grad():
                    non_action_max_q2 = self.forwardQ2(states[failure_mask], obs[1][failure_mask], obs[0][failure_mask],
                                                       obs_encoding, pixel[failure_mask], target_net=True)
                    failure_a2_idx = a2_idx[failure_mask, 0]
                    non_action_max_q2[torch.arange(failure_mask.sum()), failure_a2_idx] = 0
                    if q1_failure_td_target == 'non_action_max_q2':
                        q1_failure_target = non_action_max_q2.max(-1)[0].clamp(0, 1)
                    else:
                        q1_failure_target = non_action_max_q2.max(-1)[0].clamp(0, 1) / 2
            else:
                q1_failure_target = torch.tensor([]).to(q_targets.device)
        else:
            raise NotImplementedError

        if td_err_measurement == 'smooth_l1' and \
                q1_failure_td_target in ['non_action_max_q2', 'half_non_action_max_q2']:
            if q1_success_target.nelement() != 0:
                q1_td_loss = F.smooth_l1_loss(q1_preds[success_mask.repeat(self.M)], q1_success_target.repeat(self.M))
            else:
                q1_td_loss = torch.tensor(0.).to(self.device)
            if q1_failure_target.nelement() != 0:
                q1_td_loss += F.smooth_l1_loss(q1_preds[failure_mask.repeat(self.M)], q1_failure_target.repeat(self.M))
        else:
            q1_td_loss = F.smooth_l1_loss(q1_preds, q_targets.repeat(self.M))

        q2_td_loss = F.smooth_l1_loss(q2_preds, q_targets.repeat(self.M))
        td_loss = q1_td_loss + q2_td_loss
        with torch.no_grad():
            td_error = torch.abs(q2_preds - q_targets.repeat(self.M)).reshape(batch_size, -1).mean(dim=1)

        # using q2_train_q1
        if q2_train_q1.find('top') != -1:  # topN
            topN = int(q2_train_q1[3:])
            q1_topN = q1_output.reshape(batch_size, -1).topk(topN)

            d = q1_output.size(2)
            q1_topN_idx = torch.cat(((q1_topN[1] // d).reshape(-1, 1), (q1_topN[1] % d).reshape(-1, 1)), dim=1) \
                .reshape(batch_size, topN, 2)

            q1_topN_target = []
            with torch.no_grad():
                for top_i in range(1, topN):  # omit top1, since it was calculated in q1_td_loss
                    q2_topN_prime = self.forwardQ2(states, obs[1], obs[0], obs_encoding,
                                                   q1_topN_idx[:, top_i, :], target_net=True)
                    q1_topN_target.append(q2_topN_prime.max(1)[0].reshape(batch_size, 1))
            q1_topN_target = torch.cat(q1_topN_target, dim=1).clamp(0, 1)
            q1_td_loss += F.smooth_l1_loss(q1_topN[0][:, 1:].reshape(batch_size, -1),
                                           q1_topN_target.reshape(batch_size, -1))
        elif q2_train_q1 == 'NMS':
            pass
        elif q2_train_q1 == 'soft_NMS':
            pass
        elif q2_train_q1.find('Boltzmann') != -1:  # Boltzmann sampling xy actions
            BoltzmannN = int(q2_train_q1[9:])
            q1_BoltzmannN_idx, q1_BoltzmannN_1d_idx = torch_utils.argSoftmax2d(q1_output, 1, num_samples=BoltzmannN,
                                                                               return_1d_idx=True)

            q1_BoltzmannN_idx = q1_BoltzmannN_idx.reshape(batch_size, BoltzmannN, 2)
            # d = q1_output.size(2)
            # q1_BoltzmannN_idx = torch.cat(((q1_BoltzmannN[1] // d).reshape(-1, 1), (q1_BoltzmannN[1] % d).reshape(-1, 1)), dim=1) \
            #     .reshape(batch_size, topN, 2)

            q1_BoltzmannN_value = []
            q1_BoltzmannN_target = []
            with torch.no_grad():
                for Boltzmann_i in range(0, BoltzmannN):
                    q2_BoltzmannN_prime = self.forwardQ2(states, obs[1], obs[0], obs_encoding,
                                                         q1_BoltzmannN_idx[:, Boltzmann_i, :], target_net=True)
                    q1_BoltzmannN_target.append(q2_BoltzmannN_prime.max(1)[0].reshape(batch_size, -1))
            for Boltzmann_i in range(0, BoltzmannN):
                q1_BoltzmannN_value.append(q1_output.reshape(batch_size, -1)[torch.arange(batch_size),
                                                                             q1_BoltzmannN_1d_idx[:, Boltzmann_i]]
                                           .reshape(batch_size, -1))
            q1_BoltzmannN_value = torch.cat(q1_BoltzmannN_value, dim=1).clamp(0, 1)
            q1_BoltzmannN_target = torch.cat(q1_BoltzmannN_target, dim=1).clamp(0, 1)
            q1_td_loss += F.smooth_l1_loss(q1_BoltzmannN_value.reshape(batch_size, -1),
                                           q1_BoltzmannN_target.reshape(batch_size, -1))
        else:
            pass

        return td_loss, td_error
