import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


def conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )


def conv1x1(in_planes, out_planes, stride=1, dilation=1, bias=False):
    "1x1 convolution with padding"

    kernel_size = np.asarray((1, 1))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        # self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)
        return out


class ResBlock_like_EquResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super(ResBlock_like_EquResBlock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(hidden_dim)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),

        )
        self.relu = nn.ReLU(hidden_dim)

        self.upscale = None
        if input_channels != hidden_dim:
            self.upscale = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            )

    def forward(self, xx):
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out


class Interpolate(nn.Module):
    def __init__(
            self,
            size=None,
            scale_factor=None,
            mode="bilinear",
            align_corners=None,
    ):
        super().__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners
        )


class InHandConv(nn.Module):
    def __init__(self, patch_shape):
        super().__init__()
        self.in_hand_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 64, kernel_size=3)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(64, 128, kernel_size=3)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_pool2', nn.MaxPool2d(2)),
            ('cnn_conv3', nn.Conv2d(128, 256, kernel_size=3)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))

    def forward(self, in_hand):
        return self.in_hand_conv(in_hand)


class ResUBase:
    def __init__(self, n_input_channel=1):
        self.conv_down_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc-conv0",
                        nn.Conv2d(
                            n_input_channel,
                            32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("enc-relu0", nn.ReLU(inplace=True)),
                    (
                        'enc-res1',
                        BasicBlock(
                            32, 32,
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool2',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res2',
                        BasicBlock(
                            32, 64,
                            downsample=nn.Sequential(
                                nn.Conv2d(32, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool3',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res3',
                        BasicBlock(
                            64, 128,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_8 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool4',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res4',
                        BasicBlock(
                            128, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_16 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool5',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res5',
                        BasicBlock(
                            256, 512,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'enc-conv5',
                        nn.Conv2d(512, 256, kernel_size=1, bias=False)
                    )
                ]
            )
        )

        self.conv_up_8 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res1',
                        BasicBlock(
                            512, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'dec-conv1',
                        nn.Conv2d(256, 128, kernel_size=1, bias=False)
                    )
                ]
            )
        )
        self.conv_up_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res2',
                        BasicBlock(
                            256, 128,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'dec-conv2',
                        nn.Conv2d(128, 64, kernel_size=1, bias=False)
                    )
                ]
            )
        )
        self.conv_up_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res3',
                        BasicBlock(
                            128, 64,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'dec-conv3',
                        nn.Conv2d(64, 32, kernel_size=1, bias=False)
                    )
                ]
            )
        )
        self.conv_up_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res1',
                        BasicBlock(
                            64, 32,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 32, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )


class ResU_like_Equ_ResU(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_middle_channels=(16, 32, 64, 128), kernel_size=3, N=8):
        super().__init__()
        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]

        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.Conv2d(n_input_channel, self.l1_c, kernel_size=3, padding=1)),
            ('enc-e2relu-0', nn.ReLU()),
            ('enc-e2res-1',
             ResBlock_like_EquResBlock(self.l1_c, self.l1_c, kernel_size=kernel_size)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.MaxPool2d(2)),
            ('enc-e2res-2',
             ResBlock_like_EquResBlock(self.l1_c, self.l2_c, kernel_size=kernel_size)),
        ]))
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.MaxPool2d(2)),
            ('enc-e2res-3',
             ResBlock_like_EquResBlock(self.l2_c, self.l3_c, kernel_size=kernel_size)),
        ]))
        self.conv_down_8 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-4', nn.MaxPool2d(2)),
            ('enc-e2res-4',
             ResBlock_like_EquResBlock(self.l3_c, self.l4_c, kernel_size=kernel_size)),
        ]))
        self.conv_down_16 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-5', nn.MaxPool2d(2)),
            ('enc-e2res-5',
             ResBlock_like_EquResBlock(self.l4_c, self.l4_c, kernel_size=kernel_size)),
        ]))

        self.conv_up_8 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-1',
             ResBlock_like_EquResBlock(2 * self.l4_c, self.l3_c, kernel_size=kernel_size)),
        ]))
        self.conv_up_4 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-2',
             ResBlock_like_EquResBlock(2 * self.l3_c, self.l2_c, kernel_size=kernel_size)),
        ]))
        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-3',
             ResBlock_like_EquResBlock(2 * self.l2_c, self.l1_c, kernel_size=kernel_size)),
        ]))
        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-4',
             ResBlock_like_EquResBlock(2 * self.l1_c, n_output_channel, kernel_size=kernel_size)),
        ]))

        self.upsample_16_8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_8_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_4_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forwardEncoder(self, obs):
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        return feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16

    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16):
        concat_8 = torch.cat((feature_map_8, self.upsample_16_8(feature_map_16)), dim=1)
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat((feature_map_4, self.upsample_8_4(feature_map_up_8)), dim=1)
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat((feature_map_2, self.upsample_4_2(feature_map_up_4)), dim=1)
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat((feature_map_1, self.upsample_2_1(feature_map_up_2)), dim=1)
        feature_map_up_1 = self.conv_up_1(concat_1)

        return feature_map_up_1

    def forward(self, obs):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.forwardEncoder(obs)
        return self.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16)


class ResUSoftmax(nn.Module, ResUBase):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24),
                 domain_shape=(1, 100, 100), last_activation_softmax=True):
        super().__init__()
        ResUBase.__init__(self, n_input_channel)
        self.q_values = nn.Conv2d(32, n_primitives, kernel_size=1, stride=1)
        if last_activation_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        self.last_activation_softmax = last_activation_softmax
        self.n_primitives = n_primitives
        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, obs, in_hand):
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)

        feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8,
                                                     F.interpolate(feature_map_16, size=feature_map_8.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4,
                                                     F.interpolate(feature_map_up_8, size=feature_map_4.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2,
                                                     F.interpolate(feature_map_up_4, size=feature_map_2.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1,
                                                     F.interpolate(feature_map_up_2, size=feature_map_1.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))

        if self.last_activation_softmax:
            output_shape = torch.tensor(obs.shape)
            output_shape[1] = int(self.n_primitives / 2)
            pick_q_values = self.q_values(feature_map_up_1) \
                .reshape(obs.size(0) * int(self.n_primitives / 2), 2, -1)
            pick_q_values = self.softmax(pick_q_values)[:, 1, :]
            pick_q_values = pick_q_values.reshape(output_shape.tolist())
            q_values = pick_q_values.repeat_interleave(2, dim=1)
            # q_values = torch.cat((pick_q_values, pick_q_values), dim=1)
        else:
            q_values = self.q_values(feature_map_up_1)

        return q_values, None


class ResU_like_EquResUReg(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 128, 128), N=8,
                 df_channel=16, n_middle_channels=(16, 32, 64, 128), kernel_size=3, flip=False, quotient=False,
                 initialize=True, last_activation_softmax=True, is_fcn_si=False):
        assert n_primitives == 2
        super().__init__()
        self.last_activation_softmax = last_activation_softmax
        self.N = N
        self.df_channel = df_channel

        # the main unet path
        self.unet = ResU_like_Equ_ResU(n_input_channel=n_input_channel, n_output_channel=self.df_channel,
                                       n_middle_channels=n_middle_channels,
                                       kernel_size=kernel_size, N=N)

        if last_activation_softmax:
            self.pick_q_values = torch.nn.Sequential(
                nn.Conv2d(self.df_channel, self.df_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.df_channel, 2, kernel_size=1)
            )
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.pick_q_values = torch.nn.Sequential(
                nn.Conv2d(self.df_channel, self.df_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.df_channel, 1, kernel_size=1)
            )

    def forward(self, obs, in_hand):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.unet.forwardEncoder(obs)
        feature_map_up_1 = self.unet.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8,
                                                    feature_map_16)

        if self.last_activation_softmax:
            pick_q_values = self.pick_q_values(feature_map_up_1).reshape(obs.size(0), 2, -1)
            pick_q_values = self.softmax(pick_q_values)[:, 1, :]
            pick_q_values = pick_q_values.reshape(obs.shape[0], -1, obs.shape[2], obs.shape[3])
        else:
            pick_q_values = self.pick_q_values(feature_map_up_1)
        place_q_values = pick_q_values

        out = torch.cat((pick_q_values, place_q_values), dim=1)
        return out, 0


class CNN(nn.Module):
    def __init__(self, image_shape, n_outputs, last_activation_softmax=False):
        super().__init__()
        self.last_activation_softmax = last_activation_softmax
        self.n_outputs = n_outputs
        self.patch_conv = InHandConv(image_shape)
        conv_out_size = self._getConvOut(image_shape)
        self.fc1 = nn.Linear(conv_out_size, 1024)
        if last_activation_softmax:
            self.fc2 = nn.Linear(1024, 2 * n_outputs)
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.fc2 = nn.Linear(1024, n_outputs)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def _getConvOut(self, patch_shape):
        o1 = self.patch_conv(torch.zeros(1, *patch_shape))
        return int(np.prod(o1.size()))

    def forward(self, obs_encoding, patch):
        # obs_encoding = obs_encoding.view(obs_encoding.size(0), -1)
        # obs_encoding = obs_encoding.reshape(obs_encoding.size(0), -1)

        patch_conv_out = self.patch_conv(patch)
        patch_conv_out = patch_conv_out.view(patch.size(0), -1)

        # x = torch.cat((obs_encoding, patch_conv_out), dim=1)
        x = patch_conv_out
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.last_activation_softmax:
            x = self.softmax(x.reshape(-1, 2, self.n_outputs))[:, 1, :]
        return x


class CNN_like_EquResQ2(torch.nn.Module):
    def __init__(self, image_shape, n_rotations, n_primitives, df_channel=16, n_hidden=64,
                 last_activation_softmax=True):
        super().__init__()
        self.n_rotations = n_rotations
        self.n_primitives = n_primitives
        self.N = n_rotations * 2
        self.df_channel = df_channel

        n0 = int(n_hidden / 8)
        n1 = int(n_hidden / 4)
        n2 = int(n_hidden / 2)
        n3 = n_hidden
        self.patch_conv = torch.nn.Sequential(
            nn.Conv2d((image_shape[0] - 1), n0, kernel_size=7, padding=3),
            nn.ReLU(),

            ResBlock_like_EquResBlock(n0, n1, kernel_size=5),
            nn.MaxPool2d(2),

            ResBlock_like_EquResBlock(n1, n2, kernel_size=3),
            nn.MaxPool2d(2),

            ResBlock_like_EquResBlock(n2, n3, kernel_size=3),
            nn.MaxPool2d(2),
        )

        if last_activation_softmax:
            self.conv_2 = torch.nn.Sequential(
                nn.Conv2d(n3, 2 * n_primitives * n_rotations, kernel_size=4, padding=0),
            )
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.conv_2 = torch.nn.Sequential(
                nn.Conv2d(n3, 1 * n_primitives * n_rotations, kernel_size=4, padding=0),
            )

    def forward(self, obs_encoding, patch):
        batch_size = patch.size(0)
        patch_channel = patch.shape[1]
        image_patch = patch[:, :-1]
        patch_conv_out = self.patch_conv(image_patch)

        x = self.conv_2(patch_conv_out)
        x = x.reshape(batch_size, 2, self.n_primitives, -1)
        x = self.softmax(x)[:, 1, :]
        x = x.reshape(batch_size, self.n_primitives, -1)
        return x

# class ResU(nn.Module):
#     def __init__(self, n_input_channel=1, n_output_channel=3):
#         super(ResU, self).__init__()
#         self.conv_down_1 = nn.Sequential(OrderedDict([
#             ('enc-conv-0', nn.Conv2d(n_input_channel, 32, kernel_size=3, stride=1, padding=1)),
#             ('enc-relu-0', nn.ReLU(inplace=True)),
#             ('enc-res1', BasicBlock(32, 32))
#         ]))
#         self.conv_down_2 = nn.Sequential(OrderedDict([
#             ('enc-pool2', nn.MaxPool2d(2)),
#             ('enc-res2', BasicBlock(32, 64, downsample=nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, bias=False))))
#         ]))
#         self.conv_down_4 = nn.Sequential(OrderedDict([
#             ('enc-pool3', nn.MaxPool2d(2)),
#             ('enc-res3', BasicBlock(64, 128, downsample=nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False))))
#         ]))
#         self.conv_down_8 = nn.Sequential(OrderedDict([
#             ('enc-pool4', nn.MaxPool2d(2)),
#             ('enc-res4', BasicBlock(128, 256, downsample=nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False))))
#         ]))
#         self.conv_down_16 = nn.Sequential(OrderedDict([
#             ('enc-pool5', nn.MaxPool2d(2)),
#             (
#                 'enc-res5',
#                 BasicBlock(256, 512, downsample=nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, bias=False)))),
#             ('enc-conv5', nn.Conv2d(512, 256, kernel_size=1, bias=False))
#         ]))
#
#         self.conv_up_8 = nn.Sequential(OrderedDict([
#             (
#                 'dec-res1',
#                 BasicBlock(512, 256, downsample=nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False)))),
#             ('dec-conv1', nn.Conv2d(256, 128, kernel_size=1, bias=False))
#         ]))
#         self.conv_up_4 = nn.Sequential(OrderedDict([
#             (
#                 'dec-res2',
#                 BasicBlock(256, 128, downsample=nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False)))),
#             ('dec-conv2', nn.Conv2d(128, 64, kernel_size=1, bias=False))
#         ]))
#         self.conv_up_2 = nn.Sequential(OrderedDict([
#             ('dec-res3', BasicBlock(128, 64, downsample=nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False)))),
#             ('dec-conv3', nn.Conv2d(64, 32, kernel_size=1, bias=False))
#         ]))
#         self.conv_up_1 = nn.Sequential(OrderedDict([
#             ('dec-res4', BasicBlock(64, 32, downsample=nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, bias=False)))),
#             ('dec-conv4', nn.Conv2d(32, n_output_channel, kernel_size=1))
#         ]))
#
#     def forward(self, x):
#         feature_map_1 = self.conv_down_1(x)
#         feature_map_2 = self.conv_down_2(feature_map_1)
#         feature_map_4 = self.conv_down_4(feature_map_2)
#         feature_map_8 = self.conv_down_8(feature_map_4)
#         feature_map_16 = self.conv_down_16(feature_map_8)
#
#         feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8,
#                                                      F.interpolate(feature_map_16, size=feature_map_8.shape[-1],
#                                                                    mode='bilinear', align_corners=False)), dim=1))
#         feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4,
#                                                      F.interpolate(feature_map_up_8, size=feature_map_4.shape[-1],
#                                                                    mode='bilinear', align_corners=False)), dim=1))
#         feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2,
#                                                      F.interpolate(feature_map_up_4, size=feature_map_2.shape[-1],
#                                                                    mode='bilinear', align_corners=False)), dim=1))
#         feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1,
#                                                      F.interpolate(feature_map_up_2, size=feature_map_1.shape[-1],
#                                                                    mode='bilinear', align_corners=False)), dim=1))
#         return feature_map_up_1

#
# class ResURot(nn.Module):
#     def __init__(self, n_input_channel, n_rot=8, half_rotation=True):
#         super(ResURot, self).__init__()
#         self.feature = ResU(n_input_channel, n_output_channel=1)
#         self.n_rot = n_rot
#         if half_rotation:
#             max_rot = np.pi
#         else:
#             max_rot = 2 * np.pi
#         self.rzs = torch.from_numpy(np.linspace(0, max_rot, self.n_rot, endpoint=False)).float()
#
#         for m in self.named_modules():
#             if isinstance(m[1], nn.Conv2d):
#                 # nn.init.kaiming_normal_(m[1].weight.data)
#                 nn.init.xavier_normal_(m[1].weight.data)
#             elif isinstance(m[1], nn.BatchNorm2d):
#                 m[1].weight.data.fill_(1)
#                 m[1].bias.data.zero_()
#
#     def getAffineMatrices(self, n):
#         rotations = [self.rzs for _ in range(n)]
#         affine_mats_before = []
#         affine_mats_after = []
#         for i in range(n):
#             for rotate_theta in rotations[i]:
#                 affine_mat_before = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
#                                                 [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
#                 affine_mat_before.shape = (2, 3, 1)
#                 affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
#                 affine_mats_before.append(affine_mat_before)
#
#                 affine_mat_after = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
#                                                [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
#                 affine_mat_after.shape = (2, 3, 1)
#                 affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
#                 affine_mats_after.append(affine_mat_after)
#
#         affine_mats_before = torch.cat(affine_mats_before)
#         affine_mats_after = torch.cat(affine_mats_after)
#         return affine_mats_before, affine_mats_after
#
#     def forward(self, obs):
#         batch_size = obs.shape[0]
#         diag_length = float(obs.size(2)) * np.sqrt(2)
#         diag_length = np.ceil(diag_length / 32) * 32
#         padding_width = int((diag_length - obs.size(2)) / 2)
#
#         affine_mats_before, affine_mats_after = self.getAffineMatrices(batch_size)
#         affine_mats_before = affine_mats_before.to(obs.device)
#         affine_mats_after = affine_mats_after.to(obs.device)
#         # pad obs
#         obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
#         # expand obs into shape (n*num_rot, c, h, w)
#         obs = obs.unsqueeze(1).repeat(1, self.n_rot, 1, 1, 1)
#         obs = obs.reshape(obs.size(0) * obs.size(1), obs.size(2), obs.size(3), obs.size(4))
#         # rotate obs
#         flow_grid_before = F.affine_grid(affine_mats_before, obs.size(), align_corners=False)
#         rotated_obs = F.grid_sample(obs, flow_grid_before, mode='nearest', align_corners=False)
#         # forward network
#         conv_output = self.feature(rotated_obs)
#         # rotate output
#         flow_grid_after = F.affine_grid(affine_mats_after, conv_output.size(), align_corners=False)
#         unrotate_output = F.grid_sample(conv_output, flow_grid_after, mode='nearest', align_corners=False)
#
#         rotation_output = unrotate_output.reshape(
#             (batch_size, -1, unrotate_output.size(2), unrotate_output.size(3)))
#         predictions = rotation_output[:, :, padding_width: -padding_width, padding_width: -padding_width]
#         return predictions
