import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.filters import gaussian

from utils.parameters import args, device


class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self, agent, args):
        super(GraspModel, self).__init__()
        self.is_training = True
        self.agent = agent
        self.args = args
        self.tau = None

    def forward(self, x_in):
        raise NotImplementedError()

    def visualize(self, y_width, width_pred, y_pos, cos_pred, sin_pred, y_cos, y_sin, xc, pos_pred):
        row = torch.arange(args.input_size).reshape(-1, 1).repeat(1, args.input_size)
        col = torch.arange(args.input_size).reshape(1, -1).repeat(args.input_size, 1)
        width_visual = torch.zeros_like(y_width)
        q2_idx = width_pred * y_pos > 0
        width_visual[q2_idx] = (y_width - width_pred)[q2_idx]
        plt.figure()
        plt.imshow(y_pos[0, 0].detach().cpu())
        plt.quiver(col[q2_idx[0, 0]][0:5], row[q2_idx[0, 0]][0:5], cos_pred[0, 0][q2_idx[0, 0]][0:5].cpu(),
                   sin_pred[0, 0][q2_idx[0, 0]][0:5].cpu(), color='b', scale=5)
        plt.quiver(col[q2_idx[0, 0]][0:5], row[q2_idx[0, 0]][0:5], y_cos[0, 0][q2_idx[0, 0]][0:5].cpu(),
                   y_sin[0, 0][q2_idx[0, 0]][0:5].cpu(), color='r', scale=5)
        plt.colorbar()
        for img, name in zip([xc, y_pos, pos_pred, width_visual], ['obs', 'y_pos', 'pos_pred', 'y_width - width_pred']):
            plt.figure()
            plt.imshow(img[0, 0].detach().cpu())
            plt.title(str(name))
            plt.colorbar()
        plt.show()

    def compute_loss(self, xc, yc):
        if args.train_with_centers and self.is_training:
            y_pos, y_cos, y_sin, y_width, y_centers = yc

            pos_pred, cos_pred, sin_pred, width_pred, theta_out, width_out = self(xc, centers=y_centers)
            pos_idx = torch.zeros_like(pos_pred).bool()
            sdqfd_idx = torch.zeros_like(pos_pred).bool()
            for b_idx in range(args.batch_size):
                for num_center in range(args.q1_train_q2):
                    pos_idx[b_idx, 0, y_centers[b_idx, num_center, 0], y_centers[b_idx, num_center, 1]] = True
                    pos_pred_not_at_centers = pos_pred[b_idx] * ~pos_idx[b_idx]
                    pos_pred_at_centers = pos_pred[pos_idx][b_idx]
                sdqfd_idx[b_idx] = pos_pred_not_at_centers > (pos_pred_at_centers.max() - args.margin_l)
            # sdqfd_idx = (pos_pred * ~pos_idx) > (1 - args.margin_l)
            p_loss = F.smooth_l1_loss(pos_pred[pos_idx], torch.tensor(1.).to(device))
            if sdqfd_idx.sum() > 0:
                margin_loss = args.margin_weight * F.l1_loss(pos_pred[sdqfd_idx], torch.tensor(0.).to(device))
                p_loss = p_loss + margin_loss

            # round theta in (0, pi) to bucket(0, 7)
            theta = (torch.atan2(y_sin, y_cos) % (2 * math.pi)) / 2
            boundaries = (torch.arange(self.args.num_rotations + 1) * (math.pi / self.args.num_rotations)).to(device)
            idx_theta = torch.bucketize(theta, boundaries, right=True) - 1
            one_hot_theta = F.one_hot(idx_theta.long(), num_classes=self.args.num_rotations) \
                .permute(0, 4, 2, 3, 1).squeeze(-1).float()
            theta_out = theta_out * (y_pos.repeat(1, self.args.num_rotations, 1, 1))
            y_theta_one_hot = one_hot_theta.bool() * (theta_out > 0)
            theta_out_nc = theta_out.permute(0, 2, 3, 1).reshape(-1, self.args.num_rotations)
            theta_out_mask = theta_out_nc.sum(1) > 0
            theta_out_nc = theta_out_nc[theta_out_mask]
            idx_theta_nc = idx_theta.permute(0, 2, 3, 1).reshape(-1)[theta_out_mask]
            if len(theta_out_nc) != 0:
                cos_loss = F.cross_entropy(theta_out_nc, idx_theta_nc) / 2
            else:
                cos_loss = torch.tensor(0.).to(device)
            sin_loss = cos_loss
            width_at_one_hot = one_hot_theta * y_width.repeat(1, self.args.num_rotations, 1, 1)
            if y_theta_one_hot.bool().sum() != 0:
                width_loss = F.binary_cross_entropy(width_out[y_theta_one_hot.bool()],
                                                    width_at_one_hot[y_theta_one_hot.bool()])
            else:
                width_loss = torch.tensor(0.).to(device)

            if args.render:
                self.visualize(y_width, width_pred, y_pos, cos_pred, sin_pred, y_cos, y_sin, xc, pos_pred)

        elif args.model in ['equ_resu_nodf_flip_softmax']:
            y_pos, y_cos, y_sin, y_width = yc
            if self.is_training and self.args.train_with_y_pos:
                pos_pred, cos_pred, sin_pred, width_pred, theta_out, width_out = self(xc, y_pos=y_pos)
            else:
                pos_pred, cos_pred, sin_pred, width_pred, theta_out, width_out = self(xc)
            pos_pred = pos_pred.clamp(0, 1)
            width_pred = width_pred.clamp(0, 1)
            theta_out = theta_out.clamp(0, 1)
            width_out = width_out.clamp(0, 1)
            p_loss = F.binary_cross_entropy(pos_pred, y_pos)
            # round theta in (0, pi) to bucket(0, 7)
            theta = (torch.atan2(y_sin, y_cos) % (2 * math.pi)) / 2
            boundaries = (torch.arange(self.args.num_rotations + 1) * (math.pi / self.args.num_rotations)).to(device)
            idx_theta = torch.bucketize(theta, boundaries, right=True) - 1
            one_hot_theta = F.one_hot(idx_theta.long(), num_classes=self.args.num_rotations) \
                .permute(0, 4, 2, 3, 1).squeeze(-1).float()
            # y_pos_one_hot = one_hot_theta.bool()
            theta_out = theta_out * (y_pos.repeat(1, self.args.num_rotations, 1, 1))
            y_theta_one_hot = one_hot_theta.bool() * (theta_out > 0)
            theta_out_nc = theta_out.permute(0, 2, 3, 1).reshape(-1, self.args.num_rotations)
            theta_out_mask = theta_out_nc.sum(1) > 0
            theta_out_nc = theta_out_nc[theta_out_mask]
            idx_theta_nc = idx_theta.permute(0, 2, 3, 1).reshape(-1)[theta_out_mask]
            if len(theta_out_nc) != 0:
                cos_loss = F.cross_entropy(theta_out_nc, idx_theta_nc) / 2
                # cos_loss = F.binary_cross_entropy(theta_out_nc,
                #                                   one_hot_theta.reshape(-1, self.args.num_rotations)[theta_out_mask]) / 2
            else:
                cos_loss = torch.tensor(0.).to(device)
            sin_loss = cos_loss
            width_at_one_hot = one_hot_theta * y_width.repeat(1, self.args.num_rotations, 1, 1)
            if y_theta_one_hot.bool().sum() != 0:
                width_loss = F.binary_cross_entropy(width_out[y_theta_one_hot.bool()],
                                                    width_at_one_hot[y_theta_one_hot.bool()])
            else:
                width_loss = torch.tensor(0.).to(device)

            if args.render:
                self.visualize(y_width, width_pred, y_pos, cos_pred, sin_pred, y_cos, y_sin, xc, pos_pred)

        elif args.model in ['vpg', 'fcgqcnn']:
            y_pos, y_cos, y_sin, y_width = yc
            pos_pred, cos_pred, sin_pred, width_pred, theta_out, width_out = self(xc)
            pos_pred = pos_pred.clamp(0, 1)
            width_pred = width_pred.clamp(0, 1)
            theta_out = theta_out.clamp(0, 1)
            width_out = width_out.clamp(0, 1)
            # theta_out_at_zero = theta_out[~y_pos.bool().repeat(1, self.args.num_rotations, 1, 1)]
            # p_loss = F.binary_cross_entropy(theta_out_at_zero,
            #                                 torch.zeros_like(theta_out_at_zero).to(device))
            # p_loss = torch.tensor(0.).to(device)
            # round theta in (0, pi) to bucket(0, 7)
            theta = (torch.atan2(y_sin, y_cos) % (2 * math.pi)) / 2
            boundaries = (torch.arange(self.args.num_rotations + 1) * (math.pi / self.args.num_rotations)).to(device)
            idx_theta = torch.bucketize(theta, boundaries, right=True) - 1
            one_hot_theta = F.one_hot(idx_theta.long(), num_classes=self.args.num_rotations) \
                .permute(0, 4, 2, 3, 1).squeeze(-1).float()
            # theta_out = theta_out * (y_pos.repeat(1, self.args.num_rotations, 1, 1))
            # cos_loss = F.binary_cross_entropy(theta_out, one_hot_theta)
            valid_one_hot_theta = one_hot_theta * (y_pos.repeat(1, self.args.num_rotations, 1, 1))
            valid_one_hot_theta = valid_one_hot_theta.bool()
            # p_loss = F.binary_cross_entropy(theta_out[valid_one_hot_theta],
            #                                 torch.ones_like(theta_out[valid_one_hot_theta]))
            # cos_loss = F.binary_cross_entropy(theta_out[~valid_one_hot_theta],
            #                                   torch.zeros_like(theta_out[~valid_one_hot_theta]))
            p_loss = F.binary_cross_entropy(theta_out,
                                            y_pos.repeat(1, self.args.num_rotations, 1, 1))
            # p_loss = torch.tensor(0.).to(device)
            cos_loss = F.cross_entropy(
                theta_out.permute(0, 2, 3, 1)[y_pos.repeat(1, self.args.num_rotations, 1, 1).bool().permute(0, 2, 3, 1)]
                .reshape(-1, self.args.num_rotations),
                idx_theta[y_pos.bool()].reshape(-1))
            sin_loss = torch.tensor(0.).to(device)
            width_at_one_hot = one_hot_theta * y_width.repeat(1, self.args.num_rotations, 1, 1)
            width_loss = F.binary_cross_entropy(width_out[valid_one_hot_theta],
                                                width_at_one_hot[valid_one_hot_theta])

            if args.render:
                self.visualize(y_width, width_pred, y_pos, cos_pred, sin_pred, y_cos, y_sin, xc, pos_pred)

        else:
            y_pos, y_cos, y_sin, y_width = yc
            pos_pred, cos_pred, sin_pred, width_pred = self(xc)
            p_loss = F.smooth_l1_loss(pos_pred, y_pos)
            cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
            sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
            width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }

    def preprocess_x_in(self, x_in, batch_size, x_in_device):
        if self.args.normalize_depth:
            x_size = args.input_size
            half_x_size = x_size // 2
            row = torch.arange(x_size).reshape(-1, 1).repeat(1, x_size).to(x_in_device)
            col = torch.arange(x_size).reshape(1, -1).repeat(x_size, 1).to(x_in_device)
            for i in torch.arange(batch_size):
                drow = (x_in[i, 0, half_x_size:, :].mean() - x_in[i, 0, 0:half_x_size, :].mean()) / (half_x_size)
                dcol = (x_in[i, 0, :, half_x_size:].mean() - x_in[i, 0, :, 0:half_x_size].mean()) / (half_x_size)
                background = row * drow + col * dcol
                background = (background - background.mean()).to(x_in_device)
                x_in[i, 0] = x_in[i, 0] - x_in[i, 0].mean() - background.reshape(1, 1, x_size, x_size)
                x_in[i, 0] = x_in[i, 0] / x_in[i, 0].std()  # ToDo: check tenser.std() dim
        return x_in

    def train(self):
        self.tau = self.args.train_tau
        self.is_training = True
        self.agent.train()

    def eval(self):
        self.tau = self.args.test_tau
        self.is_training = False
        self.agent.eval()

    def parameters(self):
        return self.agent.fcn.parameters()


class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in
