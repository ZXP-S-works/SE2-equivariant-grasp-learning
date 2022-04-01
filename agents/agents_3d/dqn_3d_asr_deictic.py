import numpy as np
import torch

from agents.agents_3d.dqn_3d_asr import DQN3DASR

class DQN3DASRDeictic(DQN3DASR):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)

    def getQ2Input(self, obs, center_pixel):
        patch = []
        for rz in self.rzs:
            patch.append(self.getPatch(obs, center_pixel, torch.ones(center_pixel.size(0))*rz))
        patch = torch.cat(patch)
        return patch

    def forwardQ2(self, states, in_hand, obs, obs_encoding, pixels, target_net=False, to_cpu=False):
        obs_encoding = obs_encoding.repeat(self.num_rz, 1, 1, 1)
        in_hand = in_hand.repeat(self.num_rz, 1, 1, 1)
        patch = self.getQ2Input(obs.to(self.device), pixels.to(self.device))
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q2 = self.q2 if not target_net else self.target_q2
        q2_output = q2(obs_encoding, patch).reshape(self.num_rz, states.size(0), self.num_primitives).permute(1, 2, 0)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q2_output = q2_output.cpu()
        return q2_output