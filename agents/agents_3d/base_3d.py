import numpy as np
import torch
from agents.base_agent import BaseAgent
from utils.torch_utils import Dilation


class Base3D(BaseAgent):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size)

        self.num_rz = num_rz
        self.rzs = torch.from_numpy(np.linspace(rz_range[0], rz_range[1], num_rz)).float()
        self.dilater = Dilation(patch_size)
        self.aug = False

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        rot = plan[:, 2:3]
        states = plan[:, 3:4]
        pixel_x = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_y = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        pixel_x = torch.clamp(pixel_x, 0, self.heightmap_size - 1)
        pixel_y = torch.clamp(pixel_y, 0, self.heightmap_size - 1)
        diff = (rot.expand(-1, self.num_rz) - self.rzs).abs()
        diff2 = (diff - np.pi).abs()
        rot_id = torch.min(diff, diff2).argmin(1).unsqueeze(1)
        # rot_id = (rot.expand(-1, self.num_rz) - self.rzs).abs().argmin(1).unsqueeze(1)

        x = (pixel_x.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_y.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        rot = self.rzs[rot_id]
        actions = torch.cat((x, y, rot), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y, rot_id), dim=1)
        return action_idx, actions

    def check_in_hand_not_emtpy_dilation(self, obs, batch_idx, pixel, hm_threshold):
        patch = self.getPatch(obs[batch_idx].unsqueeze(0).to(self.device), pixel.to(self.device),
                              torch.zeros(pixel.size(0)), is_in_action_range=True)
        return self.dilater.chech_in_hand_not_emtpy_dilation(patch, patch.size(-1), hm_threshold)

    def get_positive_pixel_candidates(self, obs, batch_idx, hm_threshold):
        pixel_candidates = self.dilater.dilate(obs[batch_idx, 0, self.inward_padding:-self.inward_padding,
                                               self.inward_padding:-self.inward_padding].unsqueeze(0).unsqueeze(0),
                                               hm_threshold)
        return pixel_candidates

    def select_random_action_at_posi_pixel(self, obs, batch_idx, hm_threshold):
        pixel_candidates = self.get_positive_pixel_candidates(obs, batch_idx, hm_threshold)
        pixel_candidates = torch.nonzero(pixel_candidates.squeeze(0).squeeze(0))
        if pixel_candidates.size(0) <= 0:
            pixel_candidates = torch.nonzero(obs[batch_idx, 0, self.inward_padding:-self.inward_padding,
                                             self.inward_padding:-self.inward_padding] >= 0)
        pixel = pixel_candidates[np.random.randint(pixel_candidates.size(0))].reshape(1, 2)
        return pixel

    def _loadBatchToDevice(self, batch):
        super()._loadBatchToDevice(batch)
