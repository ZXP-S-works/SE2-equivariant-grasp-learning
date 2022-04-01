import numpy as np
import torch

from utils.parameters import num_rotations
from .grasp_model import GraspModel


class GRCONVNETGGCNN(GraspModel):

    def __init__(self, agent, args):
        super(GRCONVNETGGCNN, self).__init__(agent, args)

    def forward(self, x_in, output_input_size=True, centers=None):
        batch_size = x_in.size(0)
        device = x_in.device
        states = torch.zeros((batch_size)).to(device)
        in_hand = torch.zeros((batch_size, 1, 1, 1)).float().to(device)
        return self.agent.fcn.forward(x_in)
