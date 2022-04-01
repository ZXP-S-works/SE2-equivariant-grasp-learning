import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from agents.agents_3d.base_3d import Base3D
from utils import torch_utils
from utils.parameters import *


class DQN3DASR(Base3D):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8), network='unet'):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)
        self.num_rz = num_rz
        self.rzs = torch.from_numpy(np.linspace(rz_range[0], rz_range[1], num_rz)).float()
        self.a2_size = num_rz
        self.inward_padding = int((self.heightmap_size - action_pixel_range) / 2)
        self.q2 = None
        self.q2_optimizer = None

    def initNetwork(self, q1, q2, target_q1=None, target_q2=None):
        self.fcn = q1
        self.q2 = q2
        self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.lr, weight_decay=1e-5)
        self.networks.append(self.fcn)
        self.networks.append(self.q2)
        self.optimizers.append(self.fcn_optimizer)
        self.optimizers.append(self.q2_optimizer)

    def getQ2Input(self, obs, center_pixel):
        patch = self.getPatch(obs, center_pixel, torch.zeros(center_pixel.size(0)))
        if q2_input == 'hm' or args.use_depth == 0:
            pass
        elif q2_input == 'hm_minus_z':
            patch[:, :1, :, :] = patch[:, :1, :, :] - self.getPatch_z(patch)
        elif q2_input == 'hm_and_z':
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

        q2 = self.q2
        if args.q2_predict_width:
            q2_theta, q2_width = q2(obs_encoding, patch)
            q2_theta = q2_theta.reshape(states.size(0), self.num_primitives, -1)[
                torch.arange(0, states.size(0)),
                states.long()]
            q2_width = q2_width.reshape(states.size(0), self.num_primitives, -1)[
                torch.arange(0, states.size(0)),
                states.long()]
            if to_cpu:
                q2_theta = q2_theta.cpu()
                q2_width = q2_width.cpu()

            return q2_theta, q2_width

        else:
            q2_output = q2(obs_encoding, patch).reshape(states.size(0), self.num_primitives, -1)[
                torch.arange(0, states.size(0)),
                states.long()]
            if to_cpu:
                q2_output = q2_output.cpu()

            return q2_output

    def decodeA2(self, a2_id):
        rz_id = a2_id.reshape(a2_id.size(0), 1)
        rz = self.rzs[rz_id].reshape(a2_id.size(0), 1)
        return rz_id, rz

    def decodeActions(self, pixels, a2_id):  # ToDo
        rz_id, rz = self.decodeA2(a2_id)
        x = ((pixels[:, 0] - action_pixel_range / 2).float() * self.heightmap_resolution +
             (self.workspace[0][0] + self.workspace[0][1]) / 2).reshape(pixels.size(0), 1)
        y = ((pixels[:, 1] - action_pixel_range / 2).float() * self.heightmap_resolution +
             (self.workspace[1][0] + self.workspace[1][1]) / 2).reshape(pixels.size(0), 1)
        actions = torch.cat((x, y, rz), dim=1)
        action_idx = torch.cat((pixels, rz_id), dim=1)
        return action_idx, actions

    def getPatch(self, obs, center_pixel, rz, is_in_action_range=False):
        if is_in_action_range:  # return patch inside action range
            obs_in_action_space = obs[:, :, self.inward_padding:-self.inward_padding,
                                  self.inward_padding:-self.inward_padding]
            return super(DQN3DASR, self).getPatch(obs_in_action_space, center_pixel, rz)
        else:  # return patch include bin border
            patch_pixel = center_pixel.clone() + torch.tensor([self.inward_padding, self.inward_padding]) \
                .reshape(1, 2).to(center_pixel.device)
            return super(DQN3DASR, self).getPatch(obs, patch_pixel, rz)


    def forwardFCN(self, states, in_hand, obs, target_net=False, to_cpu=False):
        q_value_maps, obs_encoding = super(DQN3DASR, self).forwardFCN(states, in_hand, obs,
                                                                      target_net=target_net, to_cpu=to_cpu)
        if action_mask == 'square' and self.inward_padding != 0:
            q_value_maps = q_value_maps[:, self.inward_padding:-self.inward_padding,
                           self.inward_padding:-self.inward_padding]
        elif action_mask == 'square' and self.inward_padding == 0:
            pass
        elif False:
            pass  # ToDO
        else:
            raise NotImplementedError
        return q_value_maps, obs_encoding

    def getBoltzmannActions(self, states, in_hand, obs, temperature=1, eps=0,
                            is_z_heuristic=False, return_patch=False):
        with torch.no_grad():
            q_value_maps, obs_encoding = self.forwardFCN(states, in_hand, obs, to_cpu=True)
            pixels = []
            rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
            rand_mask = rand < eps

            for batch_idx in range(q_value_maps.size(0)):
                if rand_mask[batch_idx]:
                    pixel = self.select_random_action_at_posi_pixel(obs, batch_idx, hm_threshold)
                else:
                    for iteration in range(10):
                        positive_pixels = self.get_positive_pixel_candidates(obs, batch_idx, hm_threshold)
                        pixel = torch_utils.argSoftmax2d(
                            q_value_maps[batch_idx].reshape(1, 1, q_value_maps.size(-1), q_value_maps.size(-1))
                            * positive_pixels,
                            temperature)
                        # pixel = torch_utils.argSoftmax2d(
                        #     q_value_maps[batch_idx].reshape(1, 1, q_value_maps.size(-1), q_value_maps.size(-1)),
                        #     temperature)

                        if self.check_in_hand_not_emtpy_dilation(obs, batch_idx, pixel, hm_threshold):
                            break
                    if iteration == 9:
                        print('Action been selected not at positive pixel.')

                pixels.append(pixel)
            pixels = torch.cat(pixels)

            q2_output = self.forwardQ2(states, in_hand, obs, obs_encoding, pixels, to_cpu=True)
            a2_id = torch_utils.argSoftmax1d(q2_output, temperature)
            if rand_mask.sum() > 0:
                rand_a2 = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.a2_size)
                a2_id[rand_mask] = rand_a2.long().unsqueeze(1)

        if return_patch:
            patch = self.getPatch(obs.to(self.device), pixels.to(self.device), torch.zeros(pixels.size(0)))
        action_idx, actions = self.decodeActions(pixels, a2_id)

        q_value_maps = (q_value_maps, q2_output)
        if return_patch:
            return q_value_maps, action_idx, actions, patch
        return q_value_maps, action_idx, actions

    def getEGreedyActions(self, states, in_hand, obs, eps, return_patch=False):
        with torch.no_grad():
            q_value_maps, obs_encoding = self.forwardFCN(states, in_hand, obs, to_cpu=True)
            pixel_candidates = self.dilater.dilate(obs[:, :, self.inward_padding:-self.inward_padding,
                                                   self.inward_padding:-self.inward_padding],
                                                   hm_threshold) > 0
            thre_q_value_maps = q_value_maps.unsqueeze(1) * pixel_candidates
            if rand_argmax_action_top_n > 1 and action_selection == 'argmax_top_n':
                pixels = torch_utils.argmaxTopN2d(thre_q_value_maps, rand_argmax_action_top_n).long()
            else:
                pixels = torch_utils.argmax2d(thre_q_value_maps).long()

        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps

        for i, m in enumerate(rand_mask):
            if m:
                pixels[i] = self.select_random_action_at_posi_pixel(obs, i, hm_threshold)

        with torch.no_grad():
            q2_output = self.forwardQ2(states, in_hand, obs, obs_encoding, pixels, to_cpu=True)
            a2_id = torch.argmax(q2_output, 1)
        rand_a2 = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.a2_size)
        a2_id[rand_mask] = rand_a2.long()

        if return_patch:
            patch = self.getPatch(obs.to(self.device), pixels.to(self.device), torch.zeros(pixels.size(0)))
        action_idx, actions = self.decodeActions(pixels, a2_id)

        q_value_maps = (q_value_maps, q2_output)

        if return_patch:
            return q_value_maps, action_idx, actions, patch
        return q_value_maps, action_idx, actions

    def calcTDLoss(self, curiosity_l2):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pixel = action_idx[:, 0:2]
        a2_idx = action_idx[:, 2:3]
        if self.sl:
            q_target = self.gamma ** step_lefts
        elif is_bandit:
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

        self.loss_calc_dict['q1_output'] = q1_output
        self.loss_calc_dict['q2_output'] = q2_output

        success_mask = rewards.long() > 0
        failure_mask = rewards.clone() < 1
        q1_success_target = q_target[success_mask]
        # the target for q1, when grasp failed
        if q1_failure_td_target == 'rewards' or q1_failure_td_target == 'smooth_l1':
            q1_failure_target = q_target[failure_mask]
        # else:  # second_max_q2 !!! second max q2 maybe inaccurate, should use max(q2^q2[a]) instead
        elif q1_failure_td_target in ['non_action_max_q2', 'half_non_action_max_q2']:
            # non_action_max_q2: max(q2^q2[a])
            if failure_mask.sum() != 0:
                with torch.no_grad():
                    obs_failure_encoding = obs_encoding if not obs_encoding else obs_encoding[failure_mask]
                    non_action_max_q2 = self.forwardQ2(states[failure_mask], obs[1][failure_mask], obs[0][failure_mask],
                                                       obs_failure_encoding, pixel[failure_mask],
                                                       target_net=use_target_net)
                    failure_a2_idx = a2_idx[failure_mask, 0]
                    non_action_max_q2[torch.arange(failure_mask.sum()), failure_a2_idx] = 0
                    if q1_failure_td_target == 'non_action_max_q2':
                        q1_failure_target = non_action_max_q2.max(-1)[0].clamp(0, 1)
                    else:
                        q1_failure_target = non_action_max_q2.max(-1)[0].clamp(0, 1) / 2
            else:
                q1_failure_target = torch.tensor([]).to(q_target.device)
        else:
            raise NotImplementedError
        q1_success_target = q1_success_target
        q1_failure_target = q1_failure_target
        q_target = q_target

        if td_err_measurement == 'BCE':  # binary cross entropy
            q1_pred = q1_pred.clamp(0, 1)
            q2_pred = q2_pred.clamp(0, 1)
            if q1_success_target.nelement() != 0:
                q1_td_loss = F.smooth_l1_loss(q1_pred[success_mask], q1_success_target)  # loss for success grasps
            else:
                q1_td_loss = torch.tensor(0.).to(self.device)
            if q1_failure_target.nelement() != 0:
                if q1_failure_td_target == 'smooth_l1':
                    q1_td_loss += F.smooth_l1_loss(q1_pred[failure_mask], q1_failure_target)
                elif q1_failure_td_target in ['rewards', 'BCE', 'half_non_action_max_q2', 'non_action_max_q2']:
                    q1_td_loss += F.binary_cross_entropy(q1_pred[failure_mask], q1_failure_target)
                else:
                    raise NotImplementedError
            q2_td_loss = F.binary_cross_entropy(q2_pred, q_target)
        elif td_err_measurement == 'smooth_l1':  # smooth_l1
            if q1_success_target.nelement() != 0:
                q1_td_loss = F.smooth_l1_loss(q1_pred[success_mask], q1_success_target)  # loss for success grasps
            else:
                q1_td_loss = torch.tensor(0.).to(self.device)
            if q1_failure_target.nelement() != 0:
                q1_td_loss += F.smooth_l1_loss(q1_pred[failure_mask], q1_failure_target)
            q2_td_loss = F.smooth_l1_loss(q2_pred, q_target)
        elif td_err_measurement == 'q1_smooth_l1_q2_BCE':
            q1_td_loss = F.smooth_l1_loss(q1_pred, q_target)
            q2_pred = q2_pred.clamp(0, 1)
            q2_td_loss = F.binary_cross_entropy(q2_pred, q_target)
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
                                                   q1_topN_idx[:, top_i, :], target_net=use_target_net)
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
                                                         q1_BoltzmannN_idx[:, Boltzmann_i, :],
                                                         target_net=use_target_net)
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

        td_loss = q1_td_loss + q2_td_loss

        if curiosity_l2 > 0:
            q1_curiosity = F.smooth_l1_loss(q1_output, torch.ones_like(q1_output).to(q1_output.device))
            q2_curiosity = F.smooth_l1_loss(q2_output, torch.ones_like(q2_output).to(q2_output.device))
            td_loss += curiosity_l2 * (q1_curiosity + q2_curiosity)

        with torch.no_grad():
            if per_td_error == 'last':
                td_error = torch.abs(q2_pred - q_target)
            elif per_td_error == 'last_square':
                td_error = 5 * torch.abs(q2_pred - q_target)
            elif per_td_error == 'last_BCE':
                td_error = F.binary_cross_entropy(q2_pred, q_target, reduction='none')
            else:
                td_error = (torch.abs(q1_pred - q_target) + torch.abs(q2_pred - q_target)) / 2

        return td_loss, td_error

    def update(self, batch, curiosity_l2=final_curiosity_l2):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss(curiosity_l2=curiosity_l2)

        self.fcn_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        td_loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.q2.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q2_optimizer.step()

        self.loss_calc_dict = {}

        return td_loss.item(), td_error
