import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import peak_local_max

from .grasp_model import GraspModel, ResidualBlock
from utils import torch_utils
# from utils.parameters import *
from utils.parameters import heightmap_size, num_rotations


class OursMethod(GraspModel):

    def __init__(self, agent, args):
        super(OursMethod, self).__init__(agent, args)

    def forward(self, x_in, output_input_size=True, y_pos=None, centers=None):
        batch_size = x_in.size(0)
        x_size = x_in.size(-1)
        device = x_in.device
        x_in = self.preprocess_x_in(x_in, batch_size, device)

        obs = F.interpolate(x_in, (self.args.heightmap_size, self.args.heightmap_size))  #ToDO check which mode is the best
        states = torch.zeros((batch_size)).to(device)
        in_hand = torch.zeros((batch_size, 1, 1, 1)).float().to(device)
        q_value_maps, obs_encoding = self.agent.forwardFCN(states, in_hand, obs, target_net=False, to_cpu=False)
        BoltzmannN = self.args.q1_train_q2

        q_for_Boltzmann = q_value_maps.clone()
        if y_pos is not None:
            q_for_Boltzmann += F.interpolate(y_pos, (self.args.heightmap_size, self.args.heightmap_size)).squeeze(1)

        q1_BoltzmannN_idx, _ = torch_utils.argSoftmax2d(q_for_Boltzmann, self.tau,
                                                        num_samples=BoltzmannN, return_1d_idx=True)

        q1_BoltzmannN_idx = q1_BoltzmannN_idx.reshape(batch_size, BoltzmannN, 2).clip(0, self.args.heightmap_size - 1)

        cos_output = torch.zeros_like(q_value_maps)
        sin_output = torch.zeros_like(q_value_maps)
        width_output = torch.zeros_like(q_value_maps)
        theta_pre = torch.zeros((batch_size, self.args.num_rotations,
                                    self.args.heightmap_size, self.args.heightmap_size)).to(device)
        width_pre = torch.zeros((batch_size, self.args.num_rotations,
                                    self.args.heightmap_size, self.args.heightmap_size)).to(device)
        for Boltzmann_i in range(0, BoltzmannN):
            q2_BoltzmannN_prime, width = self.agent.forwardQ2(states, in_hand, obs, obs_encoding,
                                                              q1_BoltzmannN_idx[:, Boltzmann_i, :])
            q2_out = q2_BoltzmannN_prime.argmax(1).reshape(batch_size, -1)
            width_out = width[torch.arange(batch_size), q2_out.view(-1)]
            q2_theta = 2 * (q2_out * (math.pi / num_rotations) + (math.pi / (2 * num_rotations)))
            q2_theta = q2_theta.reshape(batch_size, -1)
            # notice that the GR-ConvNet uses 2 * theta
            cos_BoltzmannN = torch.cos(q2_theta)
            sin_BoltzmannN = torch.sin(q2_theta)
            for bi in range(batch_size):
                theta_pre[bi, :, q1_BoltzmannN_idx[bi, Boltzmann_i, :][0], q1_BoltzmannN_idx[bi, Boltzmann_i, :][1]]\
                    = q2_BoltzmannN_prime[bi]
                width_pre[bi, :, q1_BoltzmannN_idx[bi, Boltzmann_i, :][0], q1_BoltzmannN_idx[bi, Boltzmann_i, :][1]]\
                    = width[bi]
                cos_output[bi, q1_BoltzmannN_idx[bi, Boltzmann_i, :][0], q1_BoltzmannN_idx[bi, Boltzmann_i, :][1]] =\
                    cos_BoltzmannN[bi]
                sin_output[bi, q1_BoltzmannN_idx[bi, Boltzmann_i, :][0], q1_BoltzmannN_idx[bi, Boltzmann_i, :][1]] =\
                    sin_BoltzmannN[bi]
                width_output[bi, q1_BoltzmannN_idx[bi, Boltzmann_i, :][0], q1_BoltzmannN_idx[bi, Boltzmann_i, :][1]] =\
                    width_out[bi]

        if output_input_size:
            # mode = 'bilinear'
            q_value_maps = F.interpolate(q_value_maps.unsqueeze(1), size=(x_size, x_size))
            theta_pre = F.interpolate(theta_pre, size=(x_size, x_size))
            width_pre = F.interpolate(width_pre, size=(x_size, x_size))
            cos_output = F.interpolate(cos_output.unsqueeze(1), size=(x_size, x_size))
            sin_output = F.interpolate(sin_output.unsqueeze(1), size=(x_size, x_size))
            width_output = F.interpolate(width_output.unsqueeze(1), size=(x_size, x_size))

        # return pos_output, cos_output, sin_output, width_output
        return q_value_maps, cos_output, sin_output, width_output, theta_pre, width_pre

    def train(self):
        self.tau = self.args.train_tau
        self.is_training = True
        self.agent.train()

    def eval(self):
        self.tau = self.args.test_tau
        self.is_training = False
        self.agent.eval()

    def parameters(self):
        return list(self.agent.fcn.parameters()) + list(self.agent.q2.parameters())
