import numpy as np
import torch

from utils.parameters import num_rotations
from .grasp_model import GraspModel


class VPGFCGQCNN(GraspModel):

    def __init__(self, agent, args):
        super(VPGFCGQCNN, self).__init__(agent, args)

    def forward(self, x_in, output_input_size=True, centers=None):
        batch_size = x_in.size(0)
        x_size = x_in.size(-1)
        device = x_in.device
        x_in = self.preprocess_x_in(x_in, batch_size, device)

        states = torch.zeros((batch_size)).to(device)
        in_hand = torch.zeros((batch_size, 1, 1, 1)).float().to(device)
        q_value_maps = self.agent.forwardFCN(states, in_hand, x_in, target_net=False, to_cpu=False)
        theta_pre = q_value_maps.reshape(batch_size, num_rotations, 2, -1)[:, :, 0, :]
        width_pre = q_value_maps.reshape(batch_size, num_rotations, 2, -1)[:, :, 1, :]

        pos_output, theta_idx = torch.max(theta_pre, dim=1)
        pos_output = pos_output.reshape(batch_size, 1, self.args.input_size, self.args.input_size)
        theta_idx = theta_idx.reshape(batch_size, 1, self.args.input_size, self.args.input_size)
        # pos_output, theta_idx = torch.max(pos_output_candidate, dim=-1)
        theta = (theta_idx + 0.5) * (1 / num_rotations) * np.pi * 2
        cos_output = torch.cos(theta)
        sin_output = torch.sin(theta)
        width_output = width_pre.permute(0, 2, 1).reshape(-1, num_rotations)
        width_output = width_output[torch.arange(len(theta_idx.view(-1))), theta_idx.view(-1)]\
            .reshape(batch_size, 1, self.args.input_size, self.args.input_size)

        # return pos_output, cos_output, sin_output, width_output
        return pos_output, cos_output, sin_output, width_output, \
               theta_pre.reshape(batch_size, -1, self.args.input_size, self.args.input_size),\
               width_pre.reshape(batch_size, -1, self.args.input_size, self.args.input_size)
