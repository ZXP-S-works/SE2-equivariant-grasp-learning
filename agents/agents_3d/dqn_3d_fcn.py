import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from agents.agents_3d.base_3d import Base3D
from utils import torch_utils
from utils.parameters import action_mask, action_pixel_range, is_bandit, hm_threshold, model, model_predict_width


class DQN3DFCN(Base3D):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)

        self.inward_padding = int((self.heightmap_size - action_pixel_range) / 2)

    def initNetwork(self, fcn):
        self.fcn = fcn
        if model == 'vpg':
            self.fcn_optimizer = torch.optim.SGD(self.fcn.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        else:
            self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.networks.append(self.fcn)
        self.optimizers.append(self.fcn_optimizer)

    def getAffineMatrices(self, n, specific_rotations):
        if specific_rotations is None:
            rotations = [self.rzs for _ in range(n)]
        else:
            rotations = specific_rotations
        affine_mats_before = []
        affine_mats_after = []
        for i in range(n):
            for rotate_theta in rotations[i]:
                # counter clockwise
                affine_mat_before = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                                [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().to(self.device)
                affine_mats_before.append(affine_mat_before)

                affine_mat_after = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                               [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().to(self.device)
                affine_mats_after.append(affine_mat_after)

        affine_mats_before = torch.cat(affine_mats_before)
        affine_mats_after = torch.cat(affine_mats_after)
        return affine_mats_before, affine_mats_after

    def forwardFCN(self, states, in_hand, obs, target_net=False, to_cpu=False, specific_rotation_idxes=None):
        fcn = self.fcn
        if specific_rotation_idxes is None:
            rotations = [self.rzs for _ in range(obs.size(0))]
        else:
            rotations = self.rzs[specific_rotation_idxes]
        diag_length = float(obs.size(2)) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - obs.size(2)) / 2)
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        # pad obs
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        # expand obs into shape (n*num_rot, c, h, w)
        obs = obs.unsqueeze(1).repeat(1, len(rotations[0]), 1, 1, 1)
        in_hand = in_hand.unsqueeze(1).repeat(1, len(rotations[0]), 1, 1, 1)
        obs = obs.reshape(obs.size(0) * obs.size(1), obs.size(2), obs.size(3), obs.size(4))
        in_hand = in_hand.reshape(in_hand.size(0) * in_hand.size(1), in_hand.size(2), in_hand.size(3), in_hand.size(4))

        affine_mats_before, affine_mats_after = self.getAffineMatrices(states.size(0), specific_rotation_idxes)
        # rotate obs
        flow_grid_before = F.affine_grid(affine_mats_before, obs.size(), align_corners=False)
        rotated_obs = F.grid_sample(obs, flow_grid_before, mode='bilinear', align_corners=False)
        # forward network
        conv_output, _ = fcn(rotated_obs, in_hand)
        # rotate output
        flow_grid_after = F.affine_grid(affine_mats_after, conv_output.size(), align_corners=False)
        unrotate_output = F.grid_sample(conv_output, flow_grid_after, mode='bilinear', align_corners=False)

        if model_predict_width:
            # batch_size x num_rotations, 2, x_size, y_size
            predictions = unrotate_output
        else:
            rotation_output = unrotate_output.reshape(
                (states.shape[0], -1, unrotate_output.size(1), unrotate_output.size(2), unrotate_output.size(3)))
            rotation_output = rotation_output.permute(0, 2, 1, 3, 4)
            predictions = rotation_output[torch.arange(0, states.size(0)), states.long()]
        predictions = predictions[:, :, padding_width: -padding_width, padding_width: -padding_width]

        if action_mask == 'square' and self.inward_padding != 0:
            predictions = predictions[:, :, self.inward_padding:-self.inward_padding,
                                      self.inward_padding:-self.inward_padding]
        elif action_mask == 'square' and self.inward_padding == 0:
            pass
        elif False:
            pass  # ToDO
        else:
            raise NotImplementedError

        if to_cpu:
            predictions = predictions.cpu()
        return predictions

    def getEGreedyActions(self, states, in_hand, obs, eps, coef=0., return_patch=False):
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        with torch.no_grad():
            q_value_maps = self.forwardFCN(states, in_hand, obs, to_cpu=True)
        obs = obs.cpu()
        # thre_q_value_maps = q_value_maps * \
        #                     (obs.repeat(1, self.num_rz, 1, 1)[:, :, self.inward_padding:-self.inward_padding,
        #                      self.inward_padding:-self.inward_padding] > hm_threshold).cpu()
        pixel_candidates = self.dilater.dilate(obs[:, :, self.inward_padding:-self.inward_padding,
                                               self.inward_padding:-self.inward_padding],
                                               hm_threshold) > 0
        thre_q_value_maps = q_value_maps * pixel_candidates
        action_idx = torch_utils.argmax3d(thre_q_value_maps).long()
        pixels = action_idx[:, 1:]
        rot_idx = action_idx[:, 0:1]

        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps

        for i, m in enumerate(rand_mask):
            if m:
                pixels[i] = self.select_random_action_at_posi_pixel(obs, i, hm_threshold)

        rand_phi = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.num_rz)
        rot_idx[rand_mask, 0] = rand_phi.long()

        action_idx, actions = self.decodeActions(pixels, rot_idx)

        if return_patch:
            return q_value_maps, action_idx, actions, None
        return q_value_maps, action_idx, actions

    def decodeActions(self, pixels, rot_idx):
        rot = self.rzs[rot_idx]
        x = ((pixels[:, 0] - action_pixel_range / 2).float() * self.heightmap_resolution +
             (self.workspace[0][0] + self.workspace[0][1]) / 2).reshape(pixels.size(0), 1)
        y = ((pixels[:, 1] - action_pixel_range / 2).float() * self.heightmap_resolution +
             (self.workspace[1][0] + self.workspace[1][1]) / 2).reshape(pixels.size(0), 1)
        actions = torch.cat((x, y, rot), dim=1)
        action_idx = torch.cat((pixels, rot_idx), dim=1)
        return action_idx, actions

    def getPatch(self, obs, center_pixel, rz, is_in_action_range=False):
        if is_in_action_range:  # return patch inside action range
            obs_in_action_space = obs[:, :, self.inward_padding:-self.inward_padding,
                                  self.inward_padding:-self.inward_padding]
            return super(DQN3DFCN, self).getPatch(obs_in_action_space, center_pixel, rz)
        else:  # return patch include bin border
            patch_pixel = center_pixel.clone() + torch.tensor([self.inward_padding, self.inward_padding]) \
                .reshape(1, 2).to(center_pixel.device)
            return super(DQN3DFCN, self).getPatch(obs, patch_pixel, rz)

    def getBoltzmannActions(self, states, in_hand, obs, temperature=1, eps=0,
                            is_z_heuristic=False, return_patch=False, asr_Boltzmann=False):
        with torch.no_grad():
            q_value_maps = self.forwardFCN(states, in_hand, obs, to_cpu=True)
            pixels = []
            thetas = []
            rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
            rand_mask = rand < eps

            for batch_idx in range(obs.size(0)):

                if rand_mask[batch_idx]:
                    pixel = self.select_random_action_at_posi_pixel(obs, batch_idx, hm_threshold)
                    theta = np.random.randint(0, self.num_rz)
                else:
                    for theta in range(self.num_rz):
                        for iteration in range(10):
                            pixel, theta = torch_utils.argSoftmax3d(
                                q_value_maps[batch_idx].reshape(1, self.num_rz, q_value_maps.size(-1),
                                                                q_value_maps.size(-1)), temperature)

                            if self.check_in_hand_not_emtpy_dilation(obs, batch_idx, pixel, hm_threshold):
                                break
                            if iteration == 9:
                                print('Action been selected not at positive pixel.')

                pixels.append(pixel)
                thetas.append(torch.tensor(theta))

            pixels = torch.cat(pixels)
            a2_id = torch.tensor(thetas).reshape(q_value_maps.size(0), -1)

            # if eps is not None and eps != 0:
            #     rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
            #     rand_mask = rand < eps
            #     if asr_Boltzmann == 'asr_Boltzmann':
            #         rand_q12_mask = torch.randint(0, 3, (states.size(0),))
            #         rand_q1_mask = torch.bitwise_or(rand_q12_mask == 0, rand_q12_mask == 2)
            #         rand_q2_mask = torch.bitwise_or(rand_q12_mask == 1, rand_q12_mask == 2)
            #     elif asr_Boltzmann == 'full_asr_Boltzmann':
            #         rand_q12_mask = torch.randint(0, 3, (states.size(0),))
            #         rand_q1_mask = rand_q12_mask == 0
            #         rand_q2_mask = rand_q12_mask == 1
            #     else:
            #         rand_q1_mask = rand_mask
            #         rand_q2_mask = rand_mask
            #
            #     rand_q1_mask = torch.bitwise_and(rand_q1_mask, rand_mask)
            #     rand_q2_mask = torch.bitwise_and(rand_q2_mask, rand_mask)
            #
            #     for i, m in enumerate(rand_mask):
            #         if m and rand_q1_mask[i]:
            #             # pixels[i] = sample_random_positive
            #             pixel_candidates = torch.nonzero(obs[i, 0, self.inward_padding:-self.inward_padding,
            #                                              self.inward_padding:-self.inward_padding] > hm_threshold)
            #             if pixel_candidates.size(0) <= 0:
            #                 pixel_candidates = torch.nonzero(obs[i, 0, self.inward_padding:-self.inward_padding,
            #                                                  self.inward_padding:-self.inward_padding] >= 0)
            #             rand_pixel = pixel_candidates[np.random.randint(pixel_candidates.size(0))]
            #             pixels[i] = rand_pixel
            #
            # # q2_output = self.forwardQ2(states, in_hand, obs, obs_encoding, pixels, to_cpu=True)
            # # a2_id = torch_utils.argSoftmax1d(q2_output, temperature).long()
            #
            # if eps is not None and eps != 0:
            #     if rand_q2_mask.sum() != 0:
            #         rand_a2 = torch.randint_like(torch.empty(rand_q2_mask.sum()), 0, self.num_rz)
            #         # print(a2_id[rand_q2_mask].shape, rand_a2.long().shape)
            #         a2_id[rand_q2_mask] = rand_a2.long().reshape(a2_id[rand_q2_mask].shape)

        if return_patch:
            # patch = self.getQ2Input(obs.to(self.device), pixels.to(self.device))
            patch = self.getPatch(obs.to(self.device), pixels.to(self.device), torch.zeros(pixels.size(0)))
        action_idx, actions = self.decodeActions(pixels, a2_id)

        # q_value_maps = (q_value_maps, q2_output)
        if return_patch:
            return q_value_maps, action_idx, actions, patch
        return q_value_maps, action_idx, actions


    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        if self.sl:
            q_target = self.gamma ** step_lefts
        if is_bandit:
            q_target = rewards
        else:
            with torch.no_grad():
                q_map_prime = self.forwardFCN(next_states, next_obs[1], next_obs[0], target_net=True)
                q_prime = q_map_prime.reshape((batch_size, -1)).max(1)[0]
                q_target = rewards + self.gamma * q_prime * non_final_masks

        self.loss_calc_dict['q_target'] = q_target

        q_output = self.forwardFCN(states, obs[1], obs[0])
        q_pred = q_output[torch.arange(0, batch_size), action_idx[:, 2], action_idx[:, 0], action_idx[:, 1]]

        self.loss_calc_dict['q1_output'] = q_output

        td_loss = F.smooth_l1_loss(q_pred, q_target)
        with torch.no_grad():
            td_error = torch.abs(q_pred - q_target)

        return td_loss, td_error

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.fcn_optimizer.zero_grad()
        td_loss.backward()

        for param in self.fcn.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        self.loss_calc_dict = {}

        return td_loss.item(), td_error
