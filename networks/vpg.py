from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VisualPushingGrasping(nn.Module):

    def __init__(self, n_input_channel=1, predict_width=False):  # , snapshot=None
        super(VisualPushingGrasping, self).__init__()

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        graspnet_in_channel = 1024
        self.n_input_channel = n_input_channel
        if n_input_channel == 4:
            self.grasp_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
            graspnet_in_channel = 2048

        out_c = 2 if predict_width else 1

        # Construct network branches for pushing and grasping
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(graspnet_in_channel)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(graspnet_in_channel, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, out_c, kernel_size=1, stride=1, bias=False))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

    def forward(self, obs, _):
        # if obs.size(-1) == 192:
        #     obs = F.interpolate(obs, (384, 384))  # ToDO check which mode is the best

        obs = F.interpolate(obs, (2 * obs.size(-1), 2 * obs.size(-1)))  # ToDO check which mode is the best

        # Compute intermediate features
        if self.n_input_channel == 4:
            obs_depth = obs[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            obs_color = obs[:, 1:, :, :]
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(obs_depth)
            interm_grasp_color_feat = self.grasp_color_trunk.features(obs_color)
            interm_grasp_feat = torch.cat((interm_grasp_depth_feat, interm_grasp_color_feat), dim=1)
        else:
            interm_grasp_feat = self.grasp_depth_trunk.features(obs.repeat(1, 3, 1, 1))

        # Forward pass through branches, upsample results
        # if obs.size(-1) == 384:
        #     out = nn.Upsample(scale_factor=16, mode='bilinear').forward(self.graspnet(interm_grasp_feat))
        # elif obs.size(-1) == 320 or obs.size(-1) == 160:
        #     out = nn.Upsample(scale_factor=32, mode='bilinear').forward(self.graspnet(interm_grasp_feat))

        out = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False).forward(self.graspnet(interm_grasp_feat))

        return out, None
