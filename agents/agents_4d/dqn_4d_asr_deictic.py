import numpy as np
import torch
import time
from functools import wraps

from agents.agents_4d.dqn_4d_asr import DQN4DASR
from utils.parameters import device


class DQN4DASRDeictic(DQN4DASR):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False,
                 num_primitives=1, patch_size=24, num_rz=8,
                 rz_range=(0, 7 * np.pi / 8), num_zs=16, z_range=(0.02, 0.12)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives,
                         patch_size, num_rz, rz_range, num_zs, z_range)

    # def initTMap(self):
    #     super().initTMap()
    #     self.map = cp.array(self.map)

    def getQ3Input(self, obs, center_pixel, rz):
        patch = self.getPatch(obs, center_pixel, rz)
        patch -= self.getPatch_z(patch)
        patch_size0 = patch.size(0)
        patch = patch.repeat(self.num_zs, 1, 1, 1)
        patch -= self.zs.reshape(-1, 1, 1, 1).repeat_interleave(patch_size0, dim=0).to(device)
        patch += self.zs.mean()
        return patch

    def forwardQ3(self, states, in_hand, obs, obs_encoding, pixels, a2_id, target_net=False, to_cpu=False):
        # obs_encoding = obs_encoding.repeat(self.num_zs, 1, 1, 1)
        in_hand = in_hand.repeat(self.num_zs, 1, 1, 1)

        a2_id, a2 = self.decodeA2(a2_id)
        patch = self.getQ3Input(obs.to(self.device), pixels.to(self.device), a2)
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q3 = self.q3 if not target_net else self.target_q3
        q3_output = q3(obs_encoding, patch).reshape(self.num_zs, states.size(0), self.num_primitives).permute(1, 2, 0)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q3_output = q3_output.cpu()
        return q3_output
