import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn
from collections import OrderedDict
from utils.parameters import *


class EquiResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, N, flip=False, quotient=False, initialize=True):
        super(EquiResBlock, self).__init__()

        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                rep = r2_act.quotient_repr((None, 2))
            else:
                rep = r2_act.quotient_repr(2)
        else:
            rep = r2_act.regular_repr

        feat_type_in = nn.FieldType(r2_act, input_channels * [rep])
        feat_type_hid = nn.FieldType(r2_act, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      initialize=initialize),
            nn.ReLU(feat_type_hid)
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      initialize=initialize),

        )
        self.relu = nn.ReLU(feat_type_hid)

        self.upscale = None
        if input_channels != hidden_dim:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                          initialize=initialize),
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


class CustomLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(CustomLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class conv2d(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, N, activation=True, last=False, flip=False,
                 quotient=False, initialize=True):
        super(conv2d, self).__init__()
        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                rep = r2_act.quotient_repr((None, 2))
            else:
                rep = r2_act.quotient_repr(2)
        else:
            rep = r2_act.regular_repr

        feat_type_in = nn.FieldType(r2_act, input_channels * [rep])
        if last:
            feat_type_hid = nn.FieldType(r2_act, output_channels * [r2_act.trivial_repr])
        else:
            feat_type_hid = nn.FieldType(r2_act, output_channels * [rep])

        if activation:
            self.layer = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2, initialize=initialize),
                nn.ReLU(feat_type_hid)
            )
        else:
            self.layer = nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride,
                                   padding=(kernel_size - 1) // 2, initialize=initialize)

    def forward(self, xx):
        return self.layer(xx)


class EquCNNEnc(torch.nn.Module):
    def __init__(self, input_channel, output_channel, N, kernel_size=3, out_size=8,
                 quotient=False, initialize=True, archi='maxpool'):
        assert out_size in [8, 10]
        super().__init__()
        self.input_channel = input_channel
        self.N = N
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        if quotient:
            rep = self.r2_act.quotient_repr(2)
        else:
            rep = self.r2_act.regular_repr

        n1 = int(output_channel / 4)
        n2 = int(output_channel / 2)

        if out_size == 8:
            last_padding = kernel_size // 2 - 1
        else:
            last_padding = kernel_size // 2

        if patch_size == 24:
            self.conv = torch.nn.Sequential(OrderedDict([
                ('e2conv-1', nn.R2Conv(nn.FieldType(self.r2_act, input_channel * [self.r2_act.trivial_repr]),
                                       nn.FieldType(self.r2_act, n1 * [rep]),
                                       kernel_size=kernel_size, padding=kernel_size // 2 - 1, initialize=initialize)),
                ('e2relu-1', nn.ReLU(nn.FieldType(self.r2_act, n1 * [rep]))),
                ('e2conv-2', nn.R2Conv(nn.FieldType(self.r2_act, n1 * [rep]),
                                       nn.FieldType(self.r2_act, n2 * [rep]),
                                       kernel_size=kernel_size, padding=kernel_size // 2 - 1, initialize=initialize)),
                ('e2relu-2', nn.ReLU(nn.FieldType(self.r2_act, n2 * [rep]))),
                ('e2maxpool-2', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, n2 * [rep]), 2)),
                ('e2conv-3', nn.R2Conv(nn.FieldType(self.r2_act, n2 * [rep]),
                                       nn.FieldType(self.r2_act, output_channel * [rep]),
                                       kernel_size=kernel_size, padding=last_padding, initialize=initialize)),
                ('e2relu-3', nn.ReLU(nn.FieldType(self.r2_act, output_channel * [rep]))),
            ]))
        elif patch_size == 32:
            if archi == 'maxpool':
                self.conv = torch.nn.Sequential(OrderedDict([
                    ('e2conv-1', nn.R2Conv(nn.FieldType(self.r2_act, input_channel * [self.r2_act.trivial_repr]),
                                           nn.FieldType(self.r2_act, n1 * [rep]), kernel_size=kernel_size,
                                           initialize=initialize)),
                    ('e2relu-1', nn.ReLU(nn.FieldType(self.r2_act, n1 * [rep]))),
                    ('e2conv-2', nn.R2Conv(nn.FieldType(self.r2_act, n1 * [rep]), nn.FieldType(self.r2_act, n2 * [rep]),
                                           kernel_size=kernel_size, initialize=initialize)),
                    ('e2relu-2', nn.ReLU(nn.FieldType(self.r2_act, n2 * [rep]))),
                    ('e2maxpool-2', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, n2 * [rep]), 2)),
                    ('e2conv-3', nn.R2Conv(nn.FieldType(self.r2_act, n2 * [rep]),
                                           nn.FieldType(self.r2_act, output_channel * [rep]),
                                           kernel_size=kernel_size, padding=last_padding, initialize=initialize)),
                    ('e2relu-3', nn.ReLU(nn.FieldType(self.r2_act, output_channel * [rep]))),
                ]))
            if archi == 'stride':
                self.conv = torch.nn.Sequential(OrderedDict([
                    ('e2conv-1', nn.R2Conv(nn.FieldType(self.r2_act, input_channel * [self.r2_act.trivial_repr]),
                                           nn.FieldType(self.r2_act, n1 * [rep]), kernel_size=kernel_size,
                                           initialize=initialize)),
                    ('e2relu-1', nn.ReLU(nn.FieldType(self.r2_act, n1 * [rep]))),
                    ('e2conv-2', nn.R2Conv(nn.FieldType(self.r2_act, n1 * [rep]), nn.FieldType(self.r2_act, n2 * [rep]),
                                           kernel_size=8, initialize=initialize, stride=2)),
                    ('e2relu-2', nn.ReLU(nn.FieldType(self.r2_act, n2 * [rep]))),
                    ('e2conv-3', nn.R2Conv(nn.FieldType(self.r2_act, n2 * [rep]),
                                           nn.FieldType(self.r2_act, output_channel * [rep]),
                                           kernel_size=kernel_size, padding=last_padding, initialize=initialize)),
                    ('e2relu-3', nn.ReLU(nn.FieldType(self.r2_act, output_channel * [rep]))),
                ]))
        else:
            raise NotImplementedError

    def forward(self, x):
        # x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, self.input_channel * [self.r2_act.trivial_repr]))
        return self.conv(x)


class EquResUNet(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_middle_channels=(16, 32, 64, 128), kernel_size=3, N=8,
                 flip=False, quotient=False, initialize=True):
        super().__init__()
        self.N = N
        self.quotient = quotient
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]

        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.R2Conv(nn.FieldType(self.r2_act, n_input_channel * [self.r2_act.trivial_repr]),
                                       nn.FieldType(self.r2_act, self.l1_c * [self.repr]),
                                       kernel_size=3, padding=1, initialize=initialize)),
            ('enc-e2relu-0', nn.ReLU(nn.FieldType(self.r2_act, self.l1_c * [self.repr]))),
            ('enc-e2res-1',
             EquiResBlock(self.l1_c, self.l1_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), 2)),
            ('enc-e2res-2',
             EquiResBlock(self.l1_c, self.l2_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), 2)),
            ('enc-e2res-3',
             EquiResBlock(self.l2_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_down_8 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-4', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l3_c * [self.repr]), 2)),
            ('enc-e2res-4',
             EquiResBlock(self.l3_c, self.l4_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_down_16 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-5', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l4_c * [self.repr]), 2)),
            ('enc-e2res-5',
             EquiResBlock(self.l4_c, self.l4_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))

        self.conv_up_8 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-1',
             EquiResBlock(2 * self.l4_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_up_4 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-2',
             EquiResBlock(2 * self.l3_c, self.l2_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-3',
             EquiResBlock(2 * self.l2_c, self.l1_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-4',
             EquiResBlock(2 * self.l1_c, n_output_channel, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))

        self.upsample_16_8 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l4_c * [self.repr]), 2)
        self.upsample_8_4 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l3_c * [self.repr]), 2)
        self.upsample_4_2 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), 2)
        self.upsample_2_1 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), 2)

    def forwardEncoder(self, obs):
        obs_gt = nn.GeometricTensor(obs, nn.FieldType(self.r2_act, obs.shape[1] * [self.r2_act.trivial_repr]))
        feature_map_1 = self.conv_down_1(obs_gt)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        return feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16

    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16):
        concat_8 = torch.cat((feature_map_8.tensor, self.upsample_16_8(feature_map_16).tensor), dim=1)
        concat_8 = nn.GeometricTensor(concat_8, nn.FieldType(self.r2_act, 2 * self.l4_c * [self.repr]))
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat((feature_map_4.tensor, self.upsample_8_4(feature_map_up_8).tensor), dim=1)
        concat_4 = nn.GeometricTensor(concat_4, nn.FieldType(self.r2_act, 2 * self.l3_c * [self.repr]))
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat((feature_map_2.tensor, self.upsample_4_2(feature_map_up_4).tensor), dim=1)
        concat_2 = nn.GeometricTensor(concat_2, nn.FieldType(self.r2_act, 2 * self.l2_c * [self.repr]))
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat((feature_map_1.tensor, self.upsample_2_1(feature_map_up_2).tensor), dim=1)
        concat_1 = nn.GeometricTensor(concat_1, nn.FieldType(self.r2_act, 2 * self.l1_c * [self.repr]))
        feature_map_up_1 = self.conv_up_1(concat_1)

        return feature_map_up_1

    def forward(self, obs):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.forwardEncoder(obs)
        return self.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16)


# shallow resUnet, which has aroung 42^2 reception feild
class EquResUNet2m(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_middle_channels=(16, 32, 64, 128), kernel_size=3, N=8,
                 flip=False, quotient=False, initialize=True):
        super().__init__()
        self.N = N
        self.quotient = quotient
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]

        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.R2Conv(nn.FieldType(self.r2_act, n_input_channel * [self.r2_act.trivial_repr]),
                                       nn.FieldType(self.r2_act, self.l1_c * [self.repr]),
                                       kernel_size=3, padding=1, initialize=initialize)),
            ('enc-e2relu-0', nn.ReLU(nn.FieldType(self.r2_act, self.l1_c * [self.repr]))),
            ('enc-e2res-1',
             EquiResBlock(self.l1_c, self.l1_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), 2)),
            ('enc-e2res-2',
             EquiResBlock(self.l1_c, self.l2_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), 2)),
            ('enc-e2res-3',
             EquiResBlock(self.l2_c, self.l2_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-3',
             EquiResBlock(2 * self.l2_c, self.l1_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-4',
             EquiResBlock(2 * self.l1_c, n_output_channel, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))

        self.upsample_4_2 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), 2)
        self.upsample_2_1 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), 2)

    def forwardEncoder(self, obs):
        obs_gt = nn.GeometricTensor(obs, nn.FieldType(self.r2_act, obs.shape[1] * [self.r2_act.trivial_repr]))
        feature_map_1 = self.conv_down_1(obs_gt)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        return feature_map_1, feature_map_2, feature_map_4

    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4):
        concat_2 = torch.cat((feature_map_2.tensor, self.upsample_4_2(feature_map_4).tensor), dim=1)
        concat_2 = nn.GeometricTensor(concat_2, nn.FieldType(self.r2_act, 2 * self.l2_c * [self.repr]))
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat((feature_map_1.tensor, self.upsample_2_1(feature_map_up_2).tensor), dim=1)
        concat_1 = nn.GeometricTensor(concat_1, nn.FieldType(self.r2_act, 2 * self.l1_c * [self.repr]))
        feature_map_up_1 = self.conv_up_1(concat_1)
        return feature_map_up_1

    def forward(self, obs):
        feature_map_1, feature_map_2, feature_map_4 = self.forwardEncoder(obs)
        return self.forwardDecoder(feature_map_1, feature_map_2, feature_map_4)


# large shallow resUnet, which has aroung 42^2 reception feild
class EquResUNet2ml(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_middle_channels=(32, 64, 64, 128), kernel_size=3, N=8,
                 flip=False, quotient=False, initialize=True):
        super().__init__()
        # initialize=False #ToDo
        self.N = N
        self.quotient = quotient
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]

        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.R2Conv(nn.FieldType(self.r2_act, n_input_channel * [self.r2_act.trivial_repr]),
                                       nn.FieldType(self.r2_act, self.l1_c * [self.repr]),
                                       kernel_size=3, padding=1, initialize=initialize)),
            ('enc-e2relu-0', nn.ReLU(nn.FieldType(self.r2_act, self.l1_c * [self.repr]))),
            ('enc-e2res-1',
             EquiResBlock(self.l1_c, self.l1_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), 2)),
            ('enc-e2res-2',
             EquiResBlock(self.l1_c, self.l2_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), 2)),
            ('enc-e2res-3',
             EquiResBlock(self.l2_c, self.l2_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-3',
             EquiResBlock(2 * self.l2_c, self.l1_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-4',
             EquiResBlock(2 * self.l1_c, n_output_channel, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))

        self.upsample_4_2 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), 2)
        self.upsample_2_1 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), 2)

    def forwardEncoder(self, obs):
        obs_gt = nn.GeometricTensor(obs, nn.FieldType(self.r2_act, obs.shape[1] * [self.r2_act.trivial_repr]))
        feature_map_1 = self.conv_down_1(obs_gt)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        return feature_map_1, feature_map_2, feature_map_4

    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4):
        concat_2 = torch.cat((feature_map_2.tensor, self.upsample_4_2(feature_map_4).tensor), dim=1)
        concat_2 = nn.GeometricTensor(concat_2, nn.FieldType(self.r2_act, 2 * self.l2_c * [self.repr]))
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat((feature_map_1.tensor, self.upsample_2_1(feature_map_up_2).tensor), dim=1)
        concat_1 = nn.GeometricTensor(concat_1, nn.FieldType(self.r2_act, 2 * self.l1_c * [self.repr]))
        feature_map_up_1 = self.conv_up_1(concat_1)
        return feature_map_up_1

    def forward(self, obs):
        feature_map_1, feature_map_2, feature_map_4 = self.forwardEncoder(obs)
        return self.forwardDecoder(feature_map_1, feature_map_2, feature_map_4)


# no dynamic filter
class EquResUReg(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 128, 128), N=8,
                 df_channel=16, n_middle_channels=(16, 32, 64, 128), kernel_size=3, flip=False, quotient=False,
                 initialize=True, last_activation_softmax=False, is_fcn_si=False):
        assert n_primitives == 2
        super().__init__()
        self.last_activation_softmax = last_activation_softmax
        self.N = N
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((0, 2))  # ZXP ???
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr  # ZXP ???

        self.df_channel = df_channel

        # the main unet path
        self.unet = EquResUNet(n_input_channel=n_input_channel, n_output_channel=self.df_channel,
                               n_middle_channels=n_middle_channels,
                               kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)
        if is_fcn_si:
            if flip:
                last_repr = self.r2_act.quotient_repr((None, 2))  # ZXP ???
            else:
                last_repr = self.r2_act.quotient_repr(2)
        else:
            last_repr = self.r2_act.trivial_repr

        if last_activation_softmax:
            self.pick_q_values = torch.nn.Sequential(
                conv2d(self.df_channel, self.df_channel, kernel_size=3, stride=1, N=N, last=True, flip=flip,
                       quotient=quotient, initialize=initialize),
                nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                          nn.FieldType(self.r2_act, 2 * [last_repr]), kernel_size=1,
                          initialize=initialize)
            )
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.pick_q_values = torch.nn.Sequential(
                conv2d(self.df_channel, self.df_channel, kernel_size=3, stride=1, N=N, last=True, flip=flip,
                       quotient=quotient, initialize=initialize),
                nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                          nn.FieldType(self.r2_act, 1 * [last_repr]), kernel_size=1,
                          initialize=initialize)
            )

    def forward(self, obs, in_hand):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.unet.forwardEncoder(obs)
        feature_map_up_1 = self.unet.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8,
                                                    feature_map_16)

        if self.last_activation_softmax:
            pick_q_values = self.pick_q_values(feature_map_up_1).tensor.reshape(obs.size(0), 2, -1)
            pick_q_values = self.softmax(pick_q_values)[:, 1, :]
            pick_q_values = pick_q_values.reshape(obs.shape[0], -1, obs.shape[2], obs.shape[3])
        else:
            pick_q_values = self.pick_q_values(feature_map_up_1).tensor
        place_q_values = pick_q_values

        out = torch.cat((pick_q_values, place_q_values), dim=1)
        return out, 0


# 2 maxpool u net
class EquResU2MReg(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 128, 128), N=8,
                 df_channel=16, n_middle_channels=(16, 32, 64, 128), kernel_size=3, flip=False, quotient=False,
                 initialize=True, last_activation_softmax=False):
        assert n_primitives == 2
        super().__init__()
        self.last_activation_softmax = last_activation_softmax
        self.N = N
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))  # ZXP ???
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr  # ZXP ???

        self.df_channel = df_channel

        # the main unet path
        self.unet = EquResUNet2m(n_input_channel=n_input_channel, n_output_channel=self.df_channel,
                                 n_middle_channels=n_middle_channels,
                                 kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)

        if last_activation_softmax:
            self.pick_q_values = torch.nn.Sequential(
                conv2d(self.df_channel, self.df_channel, kernel_size=3, stride=1, N=N, last=True, flip=flip,
                       quotient=quotient, initialize=initialize),
                nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                          nn.FieldType(self.r2_act, 2 * [self.r2_act.trivial_repr]), kernel_size=1,
                          initialize=initialize)
            )
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.pick_q_values = torch.nn.Sequential(
                conv2d(self.df_channel, self.df_channel, kernel_size=3, stride=1, N=N, last=True, flip=flip,
                       quotient=quotient, initialize=initialize),
                nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                          nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]), kernel_size=1,
                          initialize=initialize)
            )

    def forward(self, obs, in_hand):
        feature_map_up_1 = self.unet.forward(obs)

        if self.last_activation_softmax:
            pick_q_values = self.pick_q_values(feature_map_up_1).tensor.reshape(obs.size(0), 2, -1)
            pick_q_values = self.softmax(pick_q_values)[:, 1, :]
            pick_q_values = pick_q_values.reshape(obs.shape[0], 1, obs.shape[2], obs.shape[3])
        else:
            pick_q_values = self.pick_q_values(feature_map_up_1).tensor
        place_q_values = pick_q_values

        out = torch.cat((pick_q_values, place_q_values), dim=1)
        return out, 0


# 2 maxpool u net large
class EquResU2MLReg(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 128, 128), N=8,
                 df_channel=16, n_middle_channels=(32, 64, 128, 128), kernel_size=3, flip=False, quotient=False,
                 initialize=True, last_activation_softmax=False):
        assert n_primitives == 2
        super().__init__()
        self.last_activation_softmax = last_activation_softmax
        self.N = N
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))  # ZXP ???
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr  # ZXP ???

        self.df_channel = df_channel

        # the main unet path
        self.unet = EquResUNet2ml(n_input_channel=n_input_channel, n_output_channel=self.df_channel,
                                  n_middle_channels=n_middle_channels,
                                  kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)

        if last_activation_softmax:
            self.pick_q_values = torch.nn.Sequential(
                conv2d(self.df_channel, self.df_channel, kernel_size=3, stride=1, N=N, last=True, flip=flip,
                       quotient=quotient, initialize=initialize),
                nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                          nn.FieldType(self.r2_act, 2 * [self.r2_act.trivial_repr]), kernel_size=1,
                          initialize=initialize)
            )
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.pick_q_values = torch.nn.Sequential(
                conv2d(self.df_channel, self.df_channel, kernel_size=3, stride=1, N=N, last=True, flip=flip,
                       quotient=quotient, initialize=initialize),
                nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                          nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]), kernel_size=1,
                          initialize=initialize)
            )

    def forward(self, obs, in_hand):
        feature_map_up_1 = self.unet.forward(obs)

        if self.last_activation_softmax:
            pick_q_values = self.pick_q_values(feature_map_up_1).tensor.reshape(obs.size(0), 2, -1)
            pick_q_values = self.softmax(pick_q_values)[:, 1, :]
            pick_q_values = pick_q_values.reshape(obs.shape[0], 1, obs.shape[2], obs.shape[3])
        else:
            pick_q_values = self.pick_q_values(feature_map_up_1).tensor
        # place_q_values = self.place_q_values(place_feature)
        place_q_values = pick_q_values

        out = torch.cat((pick_q_values, place_q_values), dim=1)
        return out, 0


class EquShiftQ23(torch.nn.Module):
    def __init__(self, image_shape, n_rotations, n_primitives, kernel_size=3, df_channel=16, n_hidden=128,
                 quotient=True, last_quotient=False, out_type='index', initialize=True,
                 last_activation_softmax=True, q2_type='convolution'):
        super().__init__()
        assert out_type in ['index', 'sum']
        if last_quotient:
            assert not quotient
        self.n_rotations = n_rotations
        self.n_primitives = n_primitives
        self.N = n_rotations * 2
        self.r2_act = gspaces.Rot2dOnR2(N=self.N)
        self.quotient = quotient
        self.out_type = out_type
        self.last_activation_softmax = last_activation_softmax
        if quotient:
            self.repr = self.r2_act.quotient_repr(2)
            n_weight = 0.5
        else:
            self.repr = self.r2_act.regular_repr
            n_weight = 1
        self.df_channel = df_channel

        if q2_type == 'convolution':
            self.patch_conv = torch.nn.Sequential(
                EquCNNEnc(image_shape[0] - 1, n_hidden, self.N, kernel_size=kernel_size, out_size=8, quotient=quotient,
                          initialize=initialize),
                nn.R2Conv(nn.FieldType(self.r2_act, n_hidden * [self.repr]),
                          nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]),
                          kernel_size=3, initialize=initialize),
                nn.ReLU(nn.FieldType(self.r2_act, df_channel * 2 * [self.repr])),
                nn.PointwiseMaxPool(nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]), 2),
                nn.R2Conv(nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]),
                          nn.FieldType(self.r2_act, df_channel * [self.repr]),
                          kernel_size=3, initialize=initialize),
                nn.ReLU(nn.FieldType(self.r2_act, df_channel * [self.repr])),
            )
        elif q2_type == 'convolution_last_no_maxpool':
            self.patch_conv = torch.nn.Sequential(
                EquCNNEnc(image_shape[0] - 1, n_hidden, self.N, kernel_size=kernel_size, out_size=8, quotient=quotient,
                          initialize=initialize),
                nn.R2Conv(nn.FieldType(self.r2_act, n_hidden * [self.repr]),
                          nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]),
                          kernel_size=4, initialize=initialize),
                nn.ReLU(nn.FieldType(self.r2_act, df_channel * 2 * [self.repr])),
                nn.R2Conv(nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]),
                          nn.FieldType(self.r2_act, df_channel * [self.repr]),
                          kernel_size=5, initialize=initialize),
                nn.ReLU(nn.FieldType(self.r2_act, df_channel * [self.repr])),
            )

        if last_quotient:
            output_repr = n_primitives * [self.r2_act.quotient_repr(2)]
        else:
            output_repr = n_primitives * [self.repr]
        if args.q2_predict_width:
            self.conv_21 = torch.nn.Sequential(
                nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                          nn.FieldType(self.r2_act, 2 * output_repr),
                          kernel_size=1, initialize=initialize)
            )
            self.softmax1 = torch.nn.Softmax(dim=1)
            self.conv_22 = torch.nn.Sequential(
                nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                          nn.FieldType(self.r2_act, 2 * output_repr),
                          kernel_size=1, initialize=initialize)
            )
            self.softmax2 = torch.nn.Softmax(dim=1)
        elif last_activation_softmax:
            self.conv_2 = torch.nn.Sequential(
                nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                          nn.FieldType(self.r2_act, 2 * output_repr),
                          kernel_size=1, initialize=initialize)
            )
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.conv_2 = torch.nn.Sequential(
                nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                          nn.FieldType(self.r2_act, output_repr),
                          kernel_size=1, initialize=initialize)
            )

    def forward(self, obs_encoding, patch):
        batch_size = patch.size(0)
        patch_channel = patch.shape[1]
        image_patch = patch[:, :-1]
        image_patch = nn.GeometricTensor(image_patch,
                                         nn.FieldType(self.r2_act, image_patch.shape[1] * [self.r2_act.trivial_repr]))

        patch_conv_out = self.patch_conv(image_patch)

        if args.q2_predict_width:
            x1 = self.conv_21(patch_conv_out).tensor
            x1 = x1.reshape(batch_size, 2, self.n_primitives, -1)
            x1 = self.softmax1(x1)[:, 1, :]
            x1 = x1.reshape(batch_size, self.n_primitives, -1)
            x2 = self.conv_22(patch_conv_out).tensor
            x2 = x2.reshape(batch_size, 2, self.n_primitives, -1)
            x2 = self.softmax2(x2)[:, 1, :]
            x2 = x2.reshape(batch_size, self.n_primitives, -1)
            return x1, x2
        elif self.last_activation_softmax:
            x = self.conv_2(patch_conv_out).tensor
            x = x.reshape(batch_size, 2, self.n_primitives, -1)
            x = self.softmax(x)[:, 1, :]
            x = x.reshape(batch_size, self.n_primitives, -1)
            return x


# no dynamic filter, resnet
class EquShiftQ2ResN(torch.nn.Module):
    def __init__(self, image_shape, n_rotations, n_primitives, kernel_size=3, df_channel=16, n_hidden=64,
                 quotient=True, last_quotient=False, out_type='index', initialize=True,
                 last_activation_softmax=True):
        super().__init__()
        assert out_type in ['index', 'sum']
        if last_quotient:
            assert not quotient
        self.n_rotations = n_rotations
        self.n_primitives = n_primitives
        self.N = n_rotations * 2
        self.r2_act = gspaces.Rot2dOnR2(N=self.N)
        self.quotient = quotient
        self.out_type = out_type
        self.last_activation_softmax = last_activation_softmax
        if quotient:
            self.repr = self.r2_act.quotient_repr(2)
            n_weight = 0.5
        else:
            self.repr = self.r2_act.regular_repr
            n_weight = 1
        self.df_channel = df_channel

        n0 = int(n_hidden / 8)
        n1 = int(n_hidden / 4)
        n2 = int(n_hidden / 2)
        n3 = n_hidden
        self.patch_conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, (image_shape[0] - 1) * [self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, n0 * [self.repr]),
                      kernel_size=7, padding=3, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, n0 * [self.repr])),

            EquiResBlock(n0, n1, kernel_size=5, N=self.N, flip=False, quotient=quotient,
                         initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, n1 * [self.repr]), 2),

            EquiResBlock(n1, n2, kernel_size=3, N=self.N, flip=False, quotient=quotient,
                         initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, n2 * [self.repr]), 2),

            EquiResBlock(n2, n3, kernel_size=3, N=self.N, flip=False, quotient=quotient,
                         initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, n3 * [self.repr]), 2),
        )

        if last_quotient:
            output_repr = n_primitives * [self.r2_act.quotient_repr(2)]
        else:
            output_repr = n_primitives * [self.repr]
        if last_activation_softmax and not args.q2_predict_width:
            self.conv_2 = torch.nn.Sequential(
                nn.R2Conv(nn.FieldType(self.r2_act, n3 * [self.repr]),
                          nn.FieldType(self.r2_act, 2 * output_repr),
                          kernel_size=4, padding=0, initialize=initialize),
            )
            self.softmax = torch.nn.Softmax(dim=1)
        elif args.q2_predict_width:
            self.conv_21 = torch.nn.Sequential(
                nn.R2Conv(nn.FieldType(self.r2_act, n3 * [self.repr]),
                          nn.FieldType(self.r2_act, 2 * output_repr),
                          kernel_size=4, padding=0, initialize=initialize),
            )
            self.softmax1 = torch.nn.Softmax(dim=1)
            self.conv_22 = torch.nn.Sequential(
                nn.R2Conv(nn.FieldType(self.r2_act, n3 * [self.repr]),
                          nn.FieldType(self.r2_act, 2 * output_repr),
                          kernel_size=4, padding=0, initialize=initialize),
            )
            self.softmax2 = torch.nn.Softmax(dim=1)
        else:
            self.conv_2 = torch.nn.Sequential(
                nn.R2Conv(nn.FieldType(self.r2_act, n3 * [self.repr]),
                          nn.FieldType(self.r2_act, output_repr),
                          kernel_size=4, padding=0, initialize=initialize),
            )

    def forward(self, obs_encoding, patch):
        batch_size = patch.size(0)
        patch_channel = patch.shape[1]
        image_patch = patch[:, :-1]
        image_patch = nn.GeometricTensor(image_patch,
                                         nn.FieldType(self.r2_act, image_patch.shape[1] * [self.r2_act.trivial_repr]))

        patch_conv_out = self.patch_conv(image_patch)

        if args.q2_predict_width:
            x1 = self.conv_21(patch_conv_out).tensor
            x1 = x1.reshape(batch_size, 2, self.n_primitives, -1)
            x1 = self.softmax1(x1)[:, 1, :]
            x1 = x1.reshape(batch_size, self.n_primitives, -1)
            x2 = self.conv_22(patch_conv_out).tensor
            x2 = x2.reshape(batch_size, 2, self.n_primitives, -1)
            x2 = self.softmax2(x2)[:, 1, :]
            x2 = x2.reshape(batch_size, self.n_primitives, -1)
            return x1, x2
        elif self.last_activation_softmax:
            x = self.conv_2(patch_conv_out).tensor
            x = x.reshape(batch_size, 2, self.n_primitives, -1)
            x = self.softmax(x)[:, 1, :]
            x = x.reshape(batch_size, self.n_primitives, -1)
            return x
        else:
            x = self.conv_2(patch_conv_out).tensor
            return x
