import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from agents.agents_4d.base_4d import Base4D
from utils import torch_utils
from utils.parameters import *


class DQN4DASR(Base4D):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9,
                 sl=False, num_primitives=1, patch_size=24,
                 num_rz=8, rz_range=(0, 7 * np.pi / 8), num_zs=16, z_range=(0.02, 0.12)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives,
                         patch_size, num_rz, rz_range, num_zs, z_range)
        self.num_rz = num_rz
        self.rzs = torch.from_numpy(np.linspace(rz_range[0], rz_range[1], num_rz)).float()
        self.a2_size = num_rz
        self.a3_size = num_zs
        self.inward_padding = int((self.heightmap_size - action_pixel_range) / 2)

        self.q2 = None
        self.q3 = None
        self.q2_optimizer = None
        self.q3_optimizer = None

        self.pixel_candidates = torch.nonzero(torch.ones((heightmap_size, heightmap_size)))

    def initNetwork(self, q1, q2, q3):
        self.fcn = q1
        self.q2 = q2
        self.q3 = q3
        self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.lr, weight_decay=1e-5)
        self.q3_optimizer = torch.optim.Adam(self.q3.parameters(), lr=self.lr, weight_decay=1e-5)
        self.networks.extend([self.fcn, self.q2, self.q3])
        self.optimizers.extend([self.fcn_optimizer, self.q2_optimizer, self.q3_optimizer])

    def getQ2Input(self, obs, center_pixel):
        patch = self.getPatch(obs, center_pixel, torch.zeros(center_pixel.size(0)))
        if q2_input == 'hm':
            pass
        elif q2_input == 'hm_minus_z':
            patch -= self.getPatch_z(patch)
        elif q2_input == 'hm_and_z':
            z_tensor = torch.ones_like(patch) * self.getPatch_z(patch)
            patch = torch.cat((patch, z_tensor), dim=1)
        else:
            raise NotImplementedError
        return patch

    def getQ3Input(self, obs, center_pixel, rz):
        patch = self.getPatch(obs, center_pixel, rz)
        # patch = self.normalizePatch(patch)  #ToDo ask dian???
        if q3_input == 'hm':
            pass
        elif q3_input == 'hm_minus_z':
            patch -= self.getPatch_z(patch)
        elif q3_input == 'hm_and_z':
            z_tensor = torch.ones_like(patch) * self.getPatch_z(patch)
            patch = torch.cat((patch, z_tensor), dim=1)
        else:
            raise NotImplementedError
        return patch

    def forwardQ2(self, states, in_hand, obs, obs_encoding, pixels, target_net=False, to_cpu=False):
        if q2_model.find('_cas') != -1:  # in cascade q1 q2 networks, obs_encoding is feature_map_up1
            patch = self.getQ2Input(obs_encoding.to(self.device), pixels.to(self.device))
        else:
            patch = self.getQ2Input(obs.to(self.device), pixels.to(self.device))
            patch = self.encodeInHand(patch, in_hand.to(self.device))

        # q2 = self.q2 if not target_net else self.target_q2
        q2_output = self.q2(obs_encoding, patch).reshape(states.size(0), self.num_primitives, -1)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q2_output = q2_output.cpu()
        return q2_output

    def forwardQ3(self, states, in_hand, obs, obs_encoding, pixels, a2_id, target_net=False, to_cpu=False):
        a2_id, a2 = self.decodeA2(a2_id)
        patch = self.getQ3Input(obs.to(self.device), pixels.to(self.device), a2)
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        # q3 = self.q3 if not target_net else self.target_q3
        q3_output = self.q3(obs_encoding, patch).reshape(states.size(0), self.num_primitives, self.num_zs)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q3_output = q3_output.cpu()
        return q3_output

    def decodeA2(self, a2_id):
        rz_id = a2_id.reshape(a2_id.size(0), 1)
        rz = self.rzs[rz_id].reshape(a2_id.size(0), 1)
        return rz_id, rz

    def decodeA3(self, a3_id):
        z_id = a3_id.reshape(a3_id.size(0), 1)
        z = self.zs[z_id].reshape(a3_id.size(0), 1)
        return z_id, z

    def decodeActions(self, pixels, a2_id, a3_id):
        rz_id, rz = self.decodeA2(a2_id)
        z_id, z = self.decodeA3(a3_id)

        x = ((pixels[:, 0] - action_pixel_range / 2).float() * self.heightmap_resolution +
             (self.workspace[0][0] + self.workspace[0][1]) / 2).reshape(pixels.size(0), 1)
        y = ((pixels[:, 1] - action_pixel_range / 2).float() * self.heightmap_resolution +
             (self.workspace[1][0] + self.workspace[1][1]) / 2).reshape(pixels.size(0), 1)
        actions = torch.cat((x, y, z, rz), dim=1)
        action_idx = torch.cat((pixels, z_id, rz_id), dim=1)
        return action_idx, actions

    def getPatch(self, obs, center_pixel, rz, is_in_action_range=False):
        if is_in_action_range:  # return patch inside action range
            obs_in_action_space = obs[:, :, self.inward_padding:-self.inward_padding,
                                  self.inward_padding:-self.inward_padding]
            return super(DQN4DASR, self).getPatch(obs_in_action_space, center_pixel, rz)
        else:  # return patch include bin border
            patch_pixel = center_pixel.clone() + torch.tensor([self.inward_padding, self.inward_padding]) \
                .reshape(1, 2).to(center_pixel.device)
            return super(DQN4DASR, self).getPatch(obs, patch_pixel, rz)

    def forwardFCN(self, states, in_hand, obs, target_net=False, to_cpu=False):
        q_value_maps, obs_encoding = super(DQN4DASR, self).forwardFCN(states, in_hand, obs,
                                                                      target_net=target_net, to_cpu=to_cpu)
        if action_mask == 'square':
            q_value_maps = q_value_maps[:, self.inward_padding:-self.inward_padding,
                           self.inward_padding:-self.inward_padding]
        elif False:
            pass  # ToDO
        else:
            raise NotImplementedError
        return q_value_maps, obs_encoding

    # def getPatch_z(self, patch):
    #     """
    #     :return: safe z
    #     """
    #
    #     grasp_patch = patch[:, :, :, int(self.patch_size / 2 - 4):int(self.patch_size / 2 + 4)]
    #     safe_z_pos = grasp_patch.flatten()[(-grasp_patch).flatten().argsort()[2:12]].mean()
    #
    #     return safe_z_pos

    def getBoltzmannActions(self, states, in_hand, obs, temperature=10, eps=None, is_z_heuristic=False,
                            return_patch=False, return_is_reached_button=False):
        with torch.no_grad():
            q_value_maps, obs_encoding = self.forwardFCN(states, in_hand, obs, to_cpu=True)
            # q_value_maps[(obs < hm_threshold).reshape(q_value_maps.shape)] = 0  # not very useful, due to exp(0)=1
            pixels = []
            for batch_idx in range(q_value_maps.size(0)):
                pixel = None
                if q1_random_sample:
                    pixel = \
                        torch_utils.argSoftmax2d((obs[batch_idx, 0, self.inward_padding:-self.inward_padding,
                                                  self.inward_padding:-self.inward_padding].unsqueeze(0)
                                                  > hm_threshold).float(), temperature).long()
                else:
                    for iteration in range(10):
                        pixel = torch_utils.argSoftmax2d(
                            q_value_maps[batch_idx].reshape(1, 1, q_value_maps.size(-1), q_value_maps.size(-1)),
                            temperature).long()

                        patch = self.getPatch(obs[batch_idx].unsqueeze(0).to(self.device), pixel.to(self.device),
                                              torch.zeros(pixel.size(0)), is_in_action_range=True)
                        if torch_utils.check_in_hand_not_empty(patch, patch.size(-1), hm_threshold):
                            break
                    if iteration == 9:
                        pixel = \
                            torch_utils.argSoftmax2d((obs[batch_idx, 0, self.inward_padding:-self.inward_padding,
                                                      self.inward_padding:-self.inward_padding].unsqueeze(0)
                                                      > hm_threshold).float(), temperature).long()

                pixels.append(pixel)
            pixels = torch.cat(pixels)
            # pixels = torch_utils.argSoftmax2d(q_value_maps, temperature).long()
            q2_output = self.forwardQ2(states, in_hand, obs, obs_encoding, pixels, to_cpu=True)
            a2_id = torch_utils.argSoftmax1d(q2_output, temperature).long()

            # q_value_maps = (q_value_maps, q2_output)
            is_reached_botton = []
            if is_z_heuristic:
                _, a2 = self.decodeA2(a2_id)
                patch = self.getQ2Input(obs.to(self.device), pixels.to(self.device))
                # patch = self.getPatch(obs, pixels.to(self.device), a2)
                patch_z = self.getPatch_z(patch.cpu())
                a3_id = (np.abs(self.zs.reshape(1, -1) - patch_z).argmin(axis=1))
            else:
                q3_output = self.forwardQ3(states, in_hand, obs, obs_encoding, pixels, a2_id, to_cpu=True)
                if self.networks[0].training:
                    if q3_adjustment == 'add_bias':
                        q3_biased = q3_output + 1
                    elif q3_adjustment == 'uniform_random':
                        q3_biased = torch.ones_like(q3_output)
                    elif q3_adjustment == 'none':
                        q3_biased = q3_output
                    else:
                        raise NotImplementedError
                else:
                    q3_biased = q3_output
                a3_id = torch_utils.argSoftmax1d(q3_biased, temperature).long()
                a2_id, a2 = self.decodeA2(a2_id)
                patch = self.getPatch(obs.to(self.device), pixels.to(self.device), a2)
                for i in range(len(a3_id)):
                    patch_z = self.getPatch_z(patch[i].unsqueeze(0)).cpu()
                    a_z = patch_z + self.zs[a3_id[i]]
                    if a_z < self.workspace[2, 0] + self.gripper_clearance:
                        a_z = self.workspace[2, 0] + self.gripper_clearance - patch_z
                        a3_id[i] = (np.abs(self.zs.reshape(1, -1) - a_z).argmin(axis=1))
                        is_reached_botton.append(True)
                    else:
                        is_reached_botton.append(False)

        if return_patch:
            # patch = self.getQ2Input(obs.to(self.device), pixels.to(self.device))
            patch = self.getPatch(obs.to(self.device), pixels.to(self.device), torch.zeros(pixels.size(0)))
            if not is_z_heuristic:
                a2_id, a2 = self.decodeA2(a2_id)
                patch_q3 = self.getQ3Input(obs.to(self.device), pixels.to(self.device), a2)
            patch = (patch, patch_q3) if not is_z_heuristic else patch

        action_idx, actions = self.decodeActions(pixels, a2_id, a3_id)

        q_value_maps = (q_value_maps, q2_output, q3_output) if not is_z_heuristic else (q_value_maps, q2_output)

        if return_patch:
            if not return_is_reached_button:
                return q_value_maps, action_idx, actions, patch
            else:
                return q_value_maps, action_idx, actions, patch, is_reached_botton
        return q_value_maps, action_idx, actions

    # def getEGreedyActions(self, states, in_hand, obs, eps):
    #     with torch.no_grad():
    #         q_value_maps, obs_encoding = self.forwardFCN(states, in_hand, obs, to_cpu=True)
    #         if rand_argmax_action_top_n > 1 and action_selection == 'argmax_top_n':
    #             pixels = torch_utils.argmaxTopN2d(q_value_maps, rand_argmax_action_top_n).long()
    #         else:
    #             pixels = torch_utils.argmax2d(q_value_maps).long()
    #
    #     rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
    #     rand_mask = rand < eps
    #
    #     if type(obs) is tuple:
    #         hm, ih = obs
    #     else:
    #         hm = obs
    #     for i, m in enumerate(rand_mask):
    #         if m:
    #             # pixel_candidates = torch.nonzero(hm[i, 0] > hm_threshold)
    #             # rand_pixel = pixel_candidates[np.random.randint(pixel_candidates.size(0))]
    #             # pixels[i] = rand_pixel
    #
    #             for iteration in range(50):
    #                 rand_pixel = torch.tensor((np.random.randint(hm[i, 0].size(-1)),
    #                                            np.random.randint(hm[i, 0].size(-1)))).reshape(1, -1)
    #                 # rand_pixel = hm[i, 0][np.random.randint(hm[i, 0].size(0))]
    #                 if torch_utils.check_patch_not_empty(hm[i], in_hand.size(-1), rand_pixel, hm_threshold):
    #                     break
    #             pixels[i] = rand_pixel
    #
    #     with torch.no_grad():
    #         q2_output = self.forwardQ2(states, in_hand, obs, obs_encoding, pixels, to_cpu=True)
    #         a2_id = torch.argmax(q2_output, 1)
    #     rand_a2 = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.a2_size)
    #     a2_id[rand_mask] = rand_a2.long()
    #
    #     action_idx, actions = self.decodeActions(pixels, a2_id)
    #
    #     q_value_maps = (q_value_maps, q2_output)
    #     return q_value_maps, action_idx, actions

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pixel = action_idx[:, 0:2]
        a3_idx = action_idx[:, 2:3]
        a2_idx = action_idx[:, 3:4]
        if self.sl:
            q_target = self.gamma ** step_lefts
        if is_bandit:
            q_target = rewards
        else:
            with torch.no_grad():
                q1_map_prime, obs_prime_encoding = self.forwardFCN(next_states, next_obs[1], next_obs[0],
                                                                   target_net=True)
                x_star = torch_utils.argmax2d(q1_map_prime)
                q2_prime = self.forwardQ2(next_states, next_obs[1], next_obs[0], obs_prime_encoding, x_star,
                                          target_net=True)

                q_prime = q2_prime.max(1)[0]
                q_target = rewards + self.gamma * q_prime * non_final_masks

        self.loss_calc_dict['q_target'] = q_target

        q1_output, obs_encoding = self.forwardFCN(states, obs[1], obs[0])
        q1_pred = q1_output[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]
        q2_output = self.forwardQ2(states, obs[1], obs[0], obs_encoding, pixel)
        q2_pred = q2_output[torch.arange(batch_size), a2_idx[:, 0]]
        q3_output = self.forwardQ3(states, obs[1], obs[0], obs_encoding, pixel, a2_idx)
        q3_pred = q3_output[torch.arange(batch_size), a3_idx[:, 0]]

        self.loss_calc_dict['q1_output'] = q1_output
        self.loss_calc_dict['q2_output'] = q2_output
        self.loss_calc_dict['q3_output'] = q3_output

        success_mask = rewards.long() > 0
        failure_mask = rewards.clone() < 1
        if q1_success_td_target == 'rewards':
            q1_success_target = q_target[success_mask]
        elif q1_success_td_target == 'q2':
            with torch.no_grad():
                obs_success_encoding = obs_encoding if not obs_encoding else obs_encoding[success_mask]
                success_q2 = self.forwardQ2(states[success_mask], obs[1][success_mask], obs[0][success_mask],
                                            obs_success_encoding, pixel[success_mask], target_net=True)
                success_a2_idx = a2_idx[success_mask, 0]
                q1_success_target = success_q2[torch.arange(success_mask.sum()), success_a2_idx]
        else:
            raise NotImplementedError

        # the target for q1, when grasp failed
        if q1_failure_td_target == 'rewards' or q1_failure_td_target == 'smooth_l1':
            q1_failure_target = q_target[failure_mask]
        # else:  # second_max_q2 !!! second max q2 maybe inaccurate, should use max(q2^q2[a]) instead
        elif q1_failure_td_target in ['non_action_max_q2', 'half_non_action_max_q2', 'non_action_max_q3']:
            # non_action_max_q2: max(q2^q2[a])
            if failure_mask.sum() != 0:
                if q1_failure_td_target in ['non_action_max_q2', 'half_non_action_max_q2']:
                    with torch.no_grad():
                        obs_failure_encoding = obs_encoding if not obs_encoding else obs_encoding[failure_mask]
                        non_action_max_q2 = self.forwardQ2(states[failure_mask], obs[1][failure_mask],
                                                           obs[0][failure_mask],
                                                           obs_failure_encoding, pixel[failure_mask], target_net=True)
                        failure_a2_idx = a2_idx[failure_mask, 0]
                        non_action_max_q2[torch.arange(failure_mask.sum()), failure_a2_idx] = 0
                        if q1_failure_td_target == 'non_action_max_q2':
                            q1_failure_target = non_action_max_q2.max(-1)[0].clamp(0, 1)
                        else:
                            q1_failure_target = non_action_max_q2.max(-1)[0].clamp(0, 1) / 2
                elif q1_failure_td_target == 'non_action_max_q3':
                    with torch.no_grad():
                        obs_failure_encoding = obs_encoding if not obs_encoding else obs_encoding[failure_mask]
                        non_action_max_q3 = self.forwardQ3(states[failure_mask], obs[1][failure_mask],
                                                           obs[0][failure_mask], obs_failure_encoding,
                                                           pixel[failure_mask], a2_idx[failure_mask, 0],
                                                           target_net=True)
                        failure_a3_idx = a3_idx[failure_mask, 0]
                        non_action_max_q3[torch.arange(failure_mask.sum()), failure_a3_idx] = 0
                        q1_failure_target = non_action_max_q3.max(-1)[0].clamp(0, 1)
            else:
                q1_failure_target = None
        else:
            raise NotImplementedError

        if td_err_measurement == 'smooth_l1':  # smooth_l1
            q1_td_loss = F.smooth_l1_loss(q1_pred[success_mask], q1_success_target)
            q1_td_loss = torch.tensor([0.]).to(device) if np.isnan(q1_td_loss.item()) else q1_td_loss
            if q1_failure_target is not None:
                q1_td_loss += F.smooth_l1_loss(q1_pred[failure_mask], q1_failure_target)
            if q1_failure_td_target == 'non_action_max_q3':
                q2_td_loss = F.smooth_l1_loss(q2_pred[success_mask], q1_success_target)
                q2_td_loss = torch.tensor([0.]).to(device) if np.isnan(q2_td_loss.item()) else q2_td_loss
                # loss for success grasps
                if q1_failure_target is not None:
                    q2_td_loss += F.smooth_l1_loss(q2_pred[failure_mask], q1_failure_target)
            else:
                q2_td_loss = F.smooth_l1_loss(q2_pred, q_target)
        else:
            raise NotImplementedError

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

        q3_td_loss = F.smooth_l1_loss(q3_pred, q_target)

        td_loss = q1_td_loss + q2_td_loss + q3_td_loss

        with torch.no_grad():
            if per_td_error == 'last':
                td_error = torch.abs(q2_pred - q_target)
            # elif per_td_error == 'last_square':
            #     td_error = 5 * torch.abs(q2_pred - q_target)
            # elif per_td_error == 'last_BCE':
            #     td_error = F.binary_cross_entropy(q2_pred, q_target, reduction='none')
            # else:
            #     td_error = (torch.abs(q1_pred - q_target) + torch.abs(q2_pred - q_target)) / 2

        return td_loss, td_error

    def update(self, batch, curiosity_dummy):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.fcn_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.q3_optimizer.zero_grad()
        td_loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.q2.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q2_optimizer.step()

        for param in self.q3.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q3_optimizer.step()

        self.loss_calc_dict = {}

        return td_loss.item(), td_error
