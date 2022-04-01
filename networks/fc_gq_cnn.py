from collections import OrderedDict
import torch.nn as nn


class FCGQCNN(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24),
                 domain_shape=(1, 100, 100), predict_width=False):
        super().__init__()

        out_c = 2 if predict_width else 1
        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(n_input_channel, 16, kernel_size=9, padding=4)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(16, 16, kernel_size=5, padding=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2)),
            ('conv3', nn.Conv2d(16, 16, kernel_size=5, padding=2)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(16, 16, kernel_size=5, padding=2)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(2)),
            ('conv5', nn.Conv2d(16, 128, kernel_size=17, padding=8)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(128, 128, kernel_size=1)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(128, n_primitives * out_c, kernel_size=1)),
            ('relu7', nn.ReLU(inplace=True)),
        ]))

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight.data)
                # nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, obs, in_hand):
        q_values = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False).forward(self.net(obs))
        return q_values, None

