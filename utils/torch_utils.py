import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
is_real_world = False
if is_real_world:
    import kornia as K
else:
    import cv2
import numpy as np
import collections

from collections import OrderedDict

from utils.parameters import action_sequence, aug_continuous_theta, action_pixel_range, heightmap_size, action_mask, \
    dilation_aperture

ExpertTransition = collections.namedtuple('ExpertTransition',
                                          'state obs action reward next_state next_obs done step_left expert')

action2obs_offset = int((heightmap_size - action_pixel_range) / 2)


def featureExtractor():
    '''Creates a CNN module used for feature extraction'''
    return nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(1, 16, kernel_size=7)),
        ('relu0', nn.ReLU(True)),
        ('pool0', nn.MaxPool2d(2)),
        ('conv1', nn.Conv2d(16, 32, kernel_size=7)),
        ('relu1', nn.ReLU(True)),
        ('pool1', nn.MaxPool2d(2)),
        ('conv2', nn.Conv2d(32, 64, kernel_size=5)),
        ('relu2', nn.ReLU(True)),
        ('pool2', nn.MaxPool2d(2))
    ]))


class TransformationMatrix(nn.Module):
    def __init__(self):
        super(TransformationMatrix, self).__init__()

        self.scale = torch.eye(3, 3)
        self.rotation = torch.eye(3, 3)
        self.translation = torch.eye(3, 3)

    def forward(self, scale, rotation, translation):
        scale_matrix = self.scale.repeat(scale.size(0), 1, 1)
        rotation_matrix = self.rotation.repeat(rotation.size(0), 1, 1)
        translation_matrix = self.translation.repeat(translation.size(0), 1, 1)

        scale_matrix[:, 0, 0] = scale[:, 0]
        scale_matrix[:, 1, 1] = scale[:, 1]

        rotation_matrix[:, 0, 0] = torch.cos(rotation)
        rotation_matrix[:, 0, 1] = -torch.sin(rotation)
        rotation_matrix[:, 1, 0] = torch.sin(rotation)
        rotation_matrix[:, 1, 1] = torch.cos(rotation)

        translation_matrix[:, 0, 2] = translation[:, 0]
        translation_matrix[:, 1, 2] = translation[:, 1]

        return torch.bmm(translation_matrix, torch.bmm(rotation_matrix, scale_matrix))

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.scale = self.scale.to(*args, **kwargs)
        self.rotation = self.rotation.to(*args, **kwargs)
        self.translation = self.translation.to(*args, **kwargs)
        return self


class WeightedHuberLoss(nn.Module):
    ''' Compute weighted Huber loss for use with Pioritized Expereince Replay '''

    def __init__(self):
        super(WeightedHuberLoss, self).__init__()

    def forward(self, input, target, weights, mask):
        batch_size = input.size(0)
        batch_loss = (torch.abs(input - target) < 1).float() * (input - target) ** 2 + \
                     (torch.abs(input - target) >= 1).float() * (torch.abs(input - target) - 0.5)
        batch_loss *= mask
        weighted_batch_loss = weights * batch_loss.view(batch_size, -1).sum(dim=1)
        weighted_loss = weighted_batch_loss.sum() / batch_size

        return weighted_loss


def clip(tensor, min, max):
    '''
  Clip the given tensor to the min and max values given

  Args:
    - tensor: PyTorch tensor to clip
    - min: List of min values to clip to
    - max: List of max values to clip to

  Returns: PyTorch tensor like given tensor clipped to bounds
  '''
    clipped_tensor = torch.zeros_like(tensor)
    for i in range(len(min)):
        clipped_tensor[:, i] = torch.max(torch.min(tensor[:, i], torch.tensor(max[i])), torch.tensor(min[i]))
    return clipped_tensor


def argmax2d(tensor):
    '''
  Find the index of the maximum value in a 2d tensor.

  Args:
    - tensor: PyTorch tensor of size (n x 1 x d x d)

  Returns: nx2 PyTorch tensor containing indexes of max values
  '''
    n = tensor.size(0)
    d = tensor.size(2)
    m = tensor.view(n, -1).argmax(1)
    return torch.cat(((m // d).view(-1, 1), (m % d).view(-1, 1)), dim=1)  # ??? m // d


def argSoftmax1d(tensor, temperature):
    """
  Find the index of the Softmax value in a 1d tensor. Prob is proportional to q/temperature

  Args:
    - tensor: PyTorch tensor of size (n x 1 x d)

  Returns: nx1 PyTorch tensor containing indexes of max values
  """
    n = tensor.size(0)
    probs = (tensor / temperature).view(n, -1).softmax(dim=-1)
    m = torch.multinomial(probs, 1)
    return m.long()


def argSoftmax2d(tensor, temperature, num_samples=1, return_1d_idx=False):
    """
  Find the index of the Softmax value in a 2d tensor. Prob is proportional to q/temperature

  Args:
    - tensor: PyTorch tensor of size (n x 1 x d x d)

  Returns: n x num_samples x 2 PyTorch tensor containing indexes of max values
  """
    n = tensor.size(0)
    d = tensor.size(2)
    probs = (tensor / temperature).view(n, -1).softmax(dim=-1)
    m = torch.multinomial(probs, num_samples)
    if not return_1d_idx:
        return torch.cat(((m // d).view(-1, 1), (m % d).view(-1, 1)), dim=1).long()
    else:
        return torch.cat(((m // d).view(-1, 1), (m % d).view(-1, 1)), dim=1).long(), m.long()


def argSoftmax3d(tensor, temperature, num_samples=1):
    """
  Find the index of the Softmax value in a 3d tensor. Prob is proportional to q/temperature

  Args:
    - tensor: PyTorch tensor of size (n x c x d x d)

  Returns:  n x num_samples x 2 PyTorch tensor containing indexes of max values at X Y
            n x num_samples x 1 PyTorch tensor containing indexes of max values at theta
  """
    n = tensor.size(0)
    c = tensor.size(1)
    d = tensor.size(2)
    probs = (tensor / temperature).view(n, -1).softmax(dim=-1)
    m = torch.multinomial(probs, num_samples)
    theta = m // (d * d)
    xy_idex = m - theta * d * d

    return torch.cat(((xy_idex // d).view(-1, 1), (xy_idex % d).view(-1, 1)), dim=1).long(), theta.view(-1, 1).long()


# def argmaxTopN2d(tensor, top_n):
#     '''
#   Select index randomly from top_n maximum values in a 2d tensor. Probability is proportional to the value.
#   Args:
#     - tensor: PyTorch tensor of size (n x 1 x d x d)
#
#   Returns: nx2 PyTorch tensor containing indexes of max values
#   '''
#     n = tensor.size(0)
#     d = tensor.size(2)
#     array = tensor.view(n, -1).numpy()
#     idx = array.argpartition(-top_n)[:, -top_n:]  # ZXP the top_n idx is NOT sorted
#     m = []
#     for i in range(n):
#         prob = array[i, idx[i]]
#         prob[prob < 0] = 0
#         prob = prob / prob.sum(-1)
#         m.append(np.random.choice(idx[i], p=prob))
#     m = torch.tensor(m)
#     return torch.cat(((m // d).view(-1, 1), (m % d).view(-1, 1)), dim=1)


def check_patch_not_empty(obs_i, patch_size, pixel, threshold):
    """
    :param obs_i:
    :param patch_size:
    :param pixel:
    :param threshold:
    :return:
    """
    patch_around_pixel = obs_i[0,
                         int((pixel[0, 0] - patch_size / 8).clamp(0, obs_i.size(-1) - 1)):
                         int((pixel[0, 0] + patch_size / 8).clamp(0, obs_i.size(-1) - 1)),
                         int((pixel[0, 1] - patch_size / 8).clamp(0, obs_i.size(-1) - 1)):
                         int((pixel[0, 1] + patch_size / 8).clamp(0, obs_i.size(-1) - 1))]
    if (patch_around_pixel > threshold).sum() > 1:
        return True
    return False


def circle_filter(size, diameter=None):
    """
    Generate an n x n circle_filter with 1 within the circle otherwise 0
    :param size:
    :param diameter:
    :return:
    """
    # cf = torch.zeros(n, n)
    xs = torch.arange(-(size - 1) / 2, (size - 1) / 2 + 1, 1).repeat(size, 1)
    ys = torch.arange(-(size - 1) / 2, (size - 1) / 2 + 1, 1).reshape(size, 1).repeat(1, size)
    diameter = (size - 2) if diameter is None else diameter
    cf = (xs.pow(2) + ys.pow(2)) <= (diameter / 2) ** 2
    # plt.figure()
    # plt.imshow(cf)
    # plt.show()
    return cf.unsqueeze(0).unsqueeze(0).float()


class Dilation:
    """
    Dilating depth images. Rather than thresholding positive pixels, dilating positive pixels by diameter.
    """

    def __init__(self, n):
        if dilation_aperture != 0:
            self.diameter = int(n / dilation_aperture)
            self.cf = circle_filter(int(n // 2) + 1, self.diameter)
            self.cf2 = circle_filter(int(n), 2 * self.diameter)
        else:
            self.diameter = 0
        self.n = int(n)

    def dilate(self, img, threshold):
        if self.diameter != 0:
            device = img.device
            out = torch.nn.functional.conv2d((img > threshold).float(),
                                             self.cf.to(device),
                                             padding=int(self.n // 4))
            out = (out > 0).float()
            # plt.figure()
            # plt.imshow((out[0, 0] > 0) * 0.1 + img[0, 0])
            # plt.colorbar()
            # plt.show()
        else:
            out = torch.ones_like(img)
        return out

    def chech_in_hand_not_emtpy_dilation(self, in_hand, in_hand_size, threshold):
        assert self.n == in_hand_size
        if self.diameter != 0:
            patch = (in_hand > threshold).float() * self.cf2.to(in_hand.device)
            # plt.figure()
            # plt.imshow(patch[0, 0].cpu() * 0.1 + in_hand[0, 0].cpu())
            # plt.colorbar()
            # plt.show()
            return (patch.sum() > 1).item()
        else:
            return True


def check_in_hand_not_empty(in_hand, in_hand_size, threshold):
    patch = in_hand[0, 0, int(3 * in_hand_size / 8):int(5 * in_hand_size / 8),
            int(3 * in_hand_size / 8):int(5 * in_hand_size / 8)]

    return ((patch > threshold).sum() > 1).item()


def argmax3d(tensor):
    n = tensor.size(0)
    c = tensor.size(1)
    d = tensor.size(2)
    m = tensor.contiguous().view(n, -1).argmax(1)
    return torch.cat(((m // (d * d)).view(-1, 1), ((m % (d * d)) // d).view(-1, 1), ((m % (d * d)) % d).view(-1, 1)),
                     dim=1)


def argmax4d(tensor):
    n = tensor.size(0)
    c1 = tensor.size(1)
    c2 = tensor.size(2)
    d = tensor.size(3)
    m = tensor.view(n, -1).argmax(1)

    d0 = (m // (d * d * c2)).view(-1, 1)
    d1 = ((m % (d * d * c2)) // (d * d)).view(-1, 1)
    d2 = (((m % (d * d * c2)) % (d * d)) // d).view(-1, 1)
    d3 = (((m % (d * d * c2)) % (d * d)) % d).view(-1, 1)

    return torch.cat((d0, d1, d2, d3), dim=1)


def softUpdate(target_net, source_net, tau):
    '''
  Move target  net to source net a small amount

  Args:
    - target_net: net to copy weights into
    - source_net: net to copy weights from
    - tau: Amount to update weights
  '''
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def hardUpdate(target_net, source_net):
    '''
  Copy all weights from source net to target net

  Args:
    - target_net: net to copy weights into
    - source_net: net to copy weights from
  '''
    target_net.load_state_dict(source_net.state_dict())


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
            torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                        dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def bbox(img, threshold=0.01):
    rows = np.any(img > threshold, axis=1)
    cols = np.any(img > threshold, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array([[1., 0., -pivot[0]],
                              [0., 1., -pivot[1]],
                              [0., 0., 1.]])
    image_t_pivot = np.array([[1., 0., pivot[0]],
                              [0., 1., pivot[1]],
                              [0., 0., 1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]],
                          [0., 0., 1.]])
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))


def get_random_image_transform_params(image_size, theta_dis_n=32, trans_sigma=-1,
                                      theta_range=None):
    if theta_range == 'small_range':
        theta_size = 2 * np.pi / theta_dis_n
        n_range = (np.pi / 6) // theta_size
        theta = np.random.choice(np.linspace(-n_range, n_range, 1, False)) * theta_size
    elif theta_range == 'set_theta_zero':
        theta = 0.
    elif theta_range is None:
        theta = np.random.choice(np.linspace(0, 2 * np.pi, theta_dis_n, False))
    else:
        raise NotImplementedError

    if aug_continuous_theta:
        theta_sigma = 2 * np.pi / (6 * theta_dis_n)  # six sigma
        theta += np.random.normal(0, theta_sigma)

    trans_sigma = action_pixel_range / 20 if trans_sigma == -1 else trans_sigma
    trans = np.random.normal(0, trans_sigma, size=2)  # [x, y]
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot


def perturb(current_image, next_image, pixels, set_theta_zero=False, theta_dis_n=32):
    """Data augmentation on images."""
    image_size = current_image.shape[:2]

    bbox_current = bbox(current_image)
    bbox_next = bbox(next_image)

    if np.any(np.array([bbox_current[1] - bbox_current[0], bbox_current[3] - bbox_current[2]]) > image_size[0] - 10):
        set_theta_zero = True

    pixels.extend([(bbox_current[0], bbox_current[2]),
                   (bbox_current[0], bbox_current[3]),
                   (bbox_current[1], bbox_current[2]),
                   (bbox_current[1], bbox_current[3]),
                   # (bbox_next[0], bbox_next[2]),
                   # (bbox_next[0], bbox_next[3]),
                   # (bbox_next[1], bbox_next[2]),
                   # (bbox_next[1], bbox_next[3]),
                   ])

    # Compute random rigid transform.
    while True:
        theta, trans, pivot = get_random_image_transform_params(image_size, theta_dis_n)
        if set_theta_zero:
            theta = 0.
        transform = get_image_transform(theta, trans, pivot)
        transform_params = theta, trans, pivot

        # Ensure pixels remain in the image after transform.
        is_valid = True
        new_pixels = []
        new_rounded_pixels = []
        for pixel in pixels:
            pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)

            rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
            rounded_pixel = np.flip(rounded_pixel)

            pixel = (transform @ pixel)[:2].squeeze()
            pixel = np.flip(pixel)

            in_fov_rounded = rounded_pixel[0] < image_size[0] and rounded_pixel[
                1] < image_size[1]
            in_fov = pixel[0] < image_size[0] and pixel[1] < image_size[1]

            is_valid = is_valid and np.all(rounded_pixel >= 0) and np.all(
                pixel >= 0) and in_fov_rounded and in_fov

            new_pixels.append(pixel)
            new_rounded_pixels.append(rounded_pixel)
        if is_valid:
            break

    new_pixels = new_pixels[:-4]
    new_rounded_pixels = new_rounded_pixels[:-4]

    # Apply rigid transform to image and pixel labels.
    current_image = cv2.warpAffine(
        current_image,
        transform[:2, :], (image_size[1], image_size[0]),
        flags=cv2.INTER_NEAREST)
    next_image = cv2.warpAffine(
        next_image,
        transform[:2, :], (image_size[1], image_size[0]),
        flags=cv2.INTER_NEAREST)

    return current_image, next_image, new_pixels, new_rounded_pixels, transform_params


def perturbBoundingAction(current_image, next_image, pixels, set_theta_zero=False, theta_dis_n=32):
    """Data augmentation on images."""
    image_size = current_image.shape[:2]
    pixels[0] += np.array((action2obs_offset, action2obs_offset))
    nppixels = np.array(pixels)
    bbox_current = [np.maximum(nppixels[:, 0] - 5, action2obs_offset)[0],
                    np.minimum(nppixels[:, 0] + 5, image_size[0] - action2obs_offset - 1)[0],
                    np.maximum(nppixels[:, 1] - 5, action2obs_offset)[0],
                    np.minimum(nppixels[:, 1] + 5, image_size[1] - action2obs_offset - 1)[0]]
    # bbox_next = bbox(next_image)

    if np.any(np.array([bbox_current[1], bbox_current[0], bbox_current[3], bbox_current[2]])
              > image_size[0] - action2obs_offset - 5) or \
            np.any(np.array([bbox_current[1], bbox_current[0], bbox_current[3], bbox_current[2]])
                   < action2obs_offset + 5):
        theta_range = 'set_theta_zero'
    elif np.any(np.array([bbox_current[1], bbox_current[0], bbox_current[3], bbox_current[2]])
                > image_size[0] - action2obs_offset - 10) or \
            np.any(np.array([bbox_current[1], bbox_current[0], bbox_current[3], bbox_current[2]])
                   < action2obs_offset + 10):
        theta_range = 'small_range'
    else:
        theta_range = None

    pixels.extend([np.array([bbox_current[0], bbox_current[2]]),
                   np.array([bbox_current[0], bbox_current[3]]),
                   np.array([bbox_current[1], bbox_current[2]]),
                   np.array([bbox_current[1], bbox_current[3]])])

    # Compute random rigid transform.
    while True:
        theta, trans, pivot = get_random_image_transform_params(image_size, theta_dis_n, theta_range=theta_range)
        # if set_theta_zero:
        #     theta = 0.
        transform = get_image_transform(theta, trans, pivot)
        transform_params = theta, trans, pivot

        # Ensure pixels remain in the image after transform.
        is_valid = True
        new_pixels = []
        new_rounded_pixels = []
        for pixel in pixels:
            pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)

            rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
            rounded_pixel = np.flip(rounded_pixel)

            pixel = (transform @ pixel)[:2].squeeze()
            pixel = np.flip(pixel)

            # ToDO
            if action_mask == 'square':
                in_fov_rounded = rounded_pixel[0] < image_size[0] - action2obs_offset and \
                                 rounded_pixel[1] < image_size[0] - action2obs_offset
                in_fov = pixel[0] < image_size[0] - action2obs_offset and \
                         pixel[1] < image_size[0] - action2obs_offset
                is_valid = is_valid and np.all(rounded_pixel >= action2obs_offset) and np.all(
                    pixel >= action2obs_offset) and in_fov_rounded and in_fov
            else:
                raise NotImplementedError

            new_pixels.append(pixel - np.array((action2obs_offset, action2obs_offset)))
            new_rounded_pixels.append(rounded_pixel - np.array((action2obs_offset, action2obs_offset)))
        if is_valid:
            break

    new_pixels = new_pixels[:-4]
    new_rounded_pixels = new_rounded_pixels[:-4]

    # Apply rigid transform to image and pixel labels.
    if is_real_world:
        # kornia/pytorch version
        transform = torch.tensor(transform[:2, :], dtype=torch.float32).unsqueeze(0)
        current_image = K.warp_affine(current_image.unsqueeze(0).unsqueeze(0),
                                      transform, (image_size[1], image_size[0]),
                                      mode='nearest', padding_mode='border', align_corners=True)
        current_image = current_image.squeeze(0).squeeze(0).numpy()
        if next_image is not None:
            next_image = K.warp_affine(next_image.unsqueeze(0).unsqueeze(0),
                                       transform, (image_size[1], image_size[0]),
                                       mode='nearest', padding_mode='border', align_corners=True)
        next_image = next_image.squeeze(0).squeeze(0).numpy()
    else:
        # cv2 version
        current_image = cv2.warpAffine(
            current_image.numpy(),
            transform[:2, :], (image_size[1], image_size[0]),
            flags=cv2.INTER_NEAREST)
        if next_image is not None:
            next_image = cv2.warpAffine(
                next_image.numpy(),
                transform[:2, :], (image_size[1], image_size[0]),
                flags=cv2.INTER_NEAREST)


    return current_image, next_image, new_pixels, new_rounded_pixels, transform_params


def perturbWithTheta(current_image, next_image, pixels, theta):
    """Data augmentation on images."""
    image_size = current_image.shape[:2]
    trans = (0, 0)
    pivot = (image_size[0] // 2, image_size[1] // 2)
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    # Ensure pixels remain in the image after transform.
    new_pixels = []
    new_rounded_pixels = []
    for pixel in pixels:
        pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)

        rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
        rounded_pixel = np.flip(rounded_pixel)

        pixel = (transform @ pixel)[:2].squeeze()
        pixel = np.flip(pixel)

        new_pixels.append(pixel)
        new_rounded_pixels.append(rounded_pixel)

    # Apply rigid transform to image and pixel labels.
    current_image = cv2.warpAffine(
        current_image,
        transform[:2, :], (image_size[1], image_size[0]),
        flags=cv2.INTER_NEAREST)
    next_image = cv2.warpAffine(
        next_image,
        transform[:2, :], (image_size[1], image_size[0]),
        flags=cv2.INTER_NEAREST)

    return current_image, next_image, new_pixels, new_rounded_pixels, transform_params


def augmentBuffer(buffer, aug_n, rzs):
    num_rz = len(rzs)
    aug_list = []
    for i, d in enumerate(buffer):
        for _ in range(aug_n):
            dtheta = rzs[1] - rzs[0]
            theta_dis_n = int(2 * np.pi / dtheta)
            obs, next_obs, _, (trans_pixel,), transform_params = perturb(d.obs[0].clone().numpy(),
                                                                         d.next_obs[0].clone().numpy(),
                                                                         [d.action[:2].clone().numpy()],
                                                                         theta_dis_n=theta_dis_n)
            action_theta = d.action[2].clone()
            trans_theta, _, _ = transform_params
            action_theta -= (trans_theta / dtheta).round().long()
            action_theta %= num_rz
            trans_action = torch.tensor([trans_pixel[0], trans_pixel[1], action_theta])
            aug_list.append(
                ExpertTransition(d.state, (torch.tensor(obs), d.obs[1]), trans_action, d.reward, d.next_state,
                                 (torch.tensor(next_obs), d.next_obs[1]), d.done, d.step_left, d.expert))
    for d in aug_list:
        buffer.add(d)


def augmentData2Buffer(buffer, d, rzs, aug_n, rotate, flip):
    num_rz = len(rzs)
    aug_list = []
    dtheta = rzs[1] - rzs[0]
    theta_dis_n = int(2 * np.pi / dtheta)
    primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: action_sequence.find(a),
                                                      ['p', 'x', 'y', 'z', 'r'])

    for _ in range(aug_n):
        obs, next_obs, _, (trans_pixel,), transform_params = \
            perturbBoundingAction(d.obs[0].clone(),
                                  d.next_obs[0].clone(),
                                  [d.action[:2].clone().numpy()],
                                  set_theta_zero=not rotate,
                                  theta_dis_n=theta_dis_n)
        action_theta = d.action[rot_idx].clone()
        trans_theta, _, _ = transform_params
        action_theta -= (trans_theta / dtheta).round().long()
        action_theta %= num_rz

        if z_idx == -1:
            trans_action = torch.tensor([trans_pixel[0], trans_pixel[1], action_theta])
        elif z_idx != -1:
            trans_action = torch.empty_like(d.action)
            trans_action[x_idx] = trans_pixel[0].item()
            trans_action[y_idx] = trans_pixel[1].item()
            trans_action[z_idx] = d.action[z_idx].clone().item()
            trans_action[rot_idx] = action_theta.item()

        if flip and np.random.random() > 0.5:
            flipped_obs = np.flip(obs, 0)
            flipped_next_obs = np.flip(next_obs, 0)
            flipped_xy = trans_pixel.copy()
            flipped_xy[0] = action_pixel_range - 1 - flipped_xy[0]
            flipped_theta = action_theta.clone()
            flipped_theta = (-flipped_theta) % num_rz
            flipped_action = torch.tensor([flipped_xy[0], flipped_xy[1], flipped_theta])
            if z_idx == -1:
                flipped_action = torch.tensor([flipped_xy[0], flipped_xy[1], flipped_theta])
            elif z_idx != -1:
                flipped_action = torch.empty_like(d.action)
                flipped_action[x_idx] = flipped_xy[0].item()
                flipped_action[y_idx] = flipped_xy[1].item()
                flipped_action[z_idx] = d.action[z_idx].clone().item()
                flipped_action[rot_idx] = flipped_theta.item()
            aug_list.append(
                ExpertTransition(d.state, (torch.tensor(flipped_obs.copy()), d.obs[1]), flipped_action, d.reward,
                                 d.next_state,
                                 (torch.tensor(flipped_next_obs.copy()), d.next_obs[1]), d.done, d.step_left, d.expert))
        else:
            aug_list.append(
                ExpertTransition(d.state, (torch.tensor(obs), d.obs[1]), trans_action, d.reward, d.next_state,
                                 (torch.tensor(next_obs), d.next_obs[1]), d.done, d.step_left, d.expert))

    # augDataSanityCheck([d], num_rz)
    # augDataSanityCheck(aug_list, num_rz)

    for aug_d in aug_list:
        buffer.add(aug_d)


def augmentData2BufferD4(buffer, d, rzs):
    num_rz = len(rzs)
    aug_list = []
    for j, rot in enumerate(np.linspace(0, 2 * np.pi, 4, endpoint=False)):
        dtheta = rzs[1] - rzs[0]
        obs, next_obs, _, (trans_pixel,), transform_params = perturbWithTheta(d.obs[0].clone().numpy(),
                                                                              d.next_obs[0].clone().numpy(),
                                                                              [d.action[:2].clone().numpy()], theta=rot)
        action_theta = d.action[2].clone()
        trans_theta, _, _ = transform_params
        action_theta -= (trans_theta / dtheta).round().long()
        action_theta %= num_rz
        trans_action = torch.tensor([trans_pixel[0], trans_pixel[1], action_theta])
        aug_list.append(ExpertTransition(d.state, (torch.tensor(obs), d.obs[1]), trans_action, d.reward, d.next_state,
                                         (torch.tensor(next_obs), d.next_obs[1]), d.done, d.step_left, d.expert))

        flipped_obs = np.flip(obs, 0)
        flipped_next_obs = np.flip(next_obs, 0)
        flipped_xy = trans_pixel.copy()
        # flipped_xy[0] = flipped_obs.shape[-1] - 1 - flipped_xy[0]
        flipped_xy[0] = action_pixel_range - 1 - flipped_xy[0]
        flipped_theta = action_theta.clone()
        flipped_theta = (num_rz - flipped_theta) % num_rz
        # flipped_theta = (-flipped_theta) % num_rz
        flipped_action = torch.tensor([flipped_xy[0], flipped_xy[1], flipped_theta])
        aug_list.append(
            ExpertTransition(d.state, (torch.tensor(flipped_obs.copy()), d.obs[1]), flipped_action, d.reward,
                             d.next_state,
                             (torch.tensor(flipped_next_obs.copy()), d.next_obs[1]), d.done, d.step_left, d.expert))

    # augDataSanityCheck(aug_list, num_rz)

    for aug_d in aug_list:
        buffer.add(aug_d)


def getDrQAugmentedTransition(obs, action_idx=None, rzs=(0, np.pi / 2), aug_type='se2'):
    theta_dis_n = 2 * np.pi // (rzs[1] - rzs[0])
    if torch.is_tensor(theta_dis_n):
        theta_dis_n = int(theta_dis_n.item())
    num_rz = len(rzs)
    heightmap_size = obs.shape[-1]
    if aug_type in ['cn', 't', 'se2']:
        if action_idx is None:
            pixels = [[0, 0]]
        else:
            pixels = [action_idx[:2]]
        if aug_type == 'cn':
            set_trans_zero = True
            set_theta_zero = False
        elif aug_type == 't':
            set_trans_zero = False
            set_theta_zero = True
        elif aug_type == 'se2':
            set_trans_zero = False
            set_theta_zero = False
        else:
            raise NotImplementedError

        # obs, next_obs, _, (trans_pixel,), transform_params = \
        #     perturbBoundingAction(d.obs[0].clone(),
        #                           d.next_obs[0].clone(),
        #                           [d.action[:2].clone().numpy()],
        #                           set_theta_zero=not rotate,
        #                           theta_dis_n=theta_dis_n)
        aug_obs, _, _, (trans_pixel,), (trans_theta, _, _) = perturbBoundingAction(obs, None, pixels,
                                                                                   set_theta_zero=set_theta_zero,
                                                                                   theta_dis_n=theta_dis_n)
        if action_idx is not None:
            action_theta = action_idx[2]
            action_theta -= (trans_theta / rzs[1] - rzs[0]).round().long()
            action_theta %= num_rz
            trans_action = [trans_pixel[0], trans_pixel[1], action_theta]
        else:
            trans_action = None
        return aug_obs, trans_action
    elif aug_type == 'shift':
        while True:
            padded_obs = np.pad(obs, [4, 4], mode='edge')
            mag_x = np.random.randint(8)
            mag_y = np.random.randint(8)
            aug_obs = padded_obs[mag_x:mag_x + heightmap_size, mag_y:mag_y + heightmap_size]
            if action_idx is None:
                trans_action = None
                break
            else:
                trans_action = [action_idx[0] - mag_x + 4, action_idx[1] - mag_y + 4, action_idx[2]]
                if (np.array(trans_action[:2]) > 0).all() and (np.array(trans_action[:2]) < heightmap_size).all():
                    break
        return aug_obs, trans_action


def augmentTransition(d, rzs, rotate=True, flip=True):
    num_rz = len(rzs)
    dtheta = rzs[1] - rzs[0]
    theta_dis_n = int(2 * np.pi / dtheta)
    primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: action_sequence.find(a),
                                                      ['p', 'x', 'y', 'z', 'r'])

    obs, next_obs, _, (trans_pixel,), transform_params = \
        perturbBoundingAction(d.obs[0].clone(),
                              d.next_obs[0].clone(),
                              [d.action[:2].clone().numpy()],
                              set_theta_zero=not rotate,
                              theta_dis_n=theta_dis_n)
    action_theta = d.action[rot_idx].clone()
    trans_theta, _, _ = transform_params
    action_theta -= (trans_theta / dtheta).round().long()
    action_theta %= num_rz

    if z_idx == -1:
        trans_action = torch.tensor([trans_pixel[0], trans_pixel[1], action_theta])
    elif z_idx != -1:
        trans_action = torch.empty_like(d.action)
        trans_action[x_idx] = trans_pixel[0].item()
        trans_action[y_idx] = trans_pixel[1].item()
        trans_action[z_idx] = d.action[z_idx].clone().item()
        trans_action[rot_idx] = action_theta.item()

    if flip and np.random.random() > 0.5:
        flipped_obs = np.flip(obs, 0)
        flipped_next_obs = np.flip(next_obs, 0)
        flipped_xy = trans_pixel.copy()
        # flipped_xy[0] = flipped_obs.shape[-1] - 1 - flipped_xy[0]
        flipped_xy[0] = action_pixel_range - 1 - flipped_xy[0]
        flipped_theta = action_theta.clone()
        flipped_theta = (-flipped_theta) % num_rz
        flipped_action = torch.tensor([flipped_xy[0], flipped_xy[1], flipped_theta])
        if z_idx == -1:
            flipped_action = torch.tensor([flipped_xy[0], flipped_xy[1], flipped_theta])
        elif z_idx != -1:
            flipped_action = torch.empty_like(d.action)
            flipped_action[x_idx] = flipped_xy[0].item()
            flipped_action[y_idx] = flipped_xy[1].item()
            flipped_action[z_idx] = d.action[z_idx].clone().item()
            flipped_action[rot_idx] = flipped_theta.item()
        aug_d = ExpertTransition(d.state, (torch.tensor(flipped_obs.copy()), d.obs[1]), flipped_action,
                                 d.reward, d.next_state, (torch.tensor(flipped_next_obs.copy()), d.next_obs[1]),
                                 d.done, d.step_left, d.expert)
    else:
        aug_d = ExpertTransition(d.state, (torch.tensor(obs), d.obs[1]), trans_action, d.reward, d.next_state,
                                 (torch.tensor(next_obs), d.next_obs[1]), d.done, d.step_left, d.expert)

    # augDataSanityCheck([d], num_rz)
    # augDataSanityCheck([aug_d], num_rz)
    return aug_d


def augDataSanityCheck(aug_list, num_rz):
    '''
  visualize augmented data (obs, action) for sanity check
  :param aug_d:
  :return:
  '''
    import matplotlib.pyplot as plt
    aug_n = len(aug_list)
    fig, axs = plt.subplots(figsize=(3 * aug_n, 3), nrows=1, ncols=aug_n)
    if aug_n == 1:
        for i, d in enumerate(aug_list):
            yaw = d.action[2].item() * np.pi / num_rz
            axs.imshow(d.obs[0])
            axs.quiver(d.action[1] + action2obs_offset, d.action[0] + action2obs_offset,
                       np.cos(yaw), np.sin(yaw), color='r', scale=10)
            axs.axis('off')
            axs.title.set_text('reward ' + str(d.reward))
    else:
        for i, d in enumerate(aug_list):
            yaw = d.action[2].item() * np.pi / num_rz
            axs[i].imshow(d.obs[0])
            axs[i].quiver(d.action[1] + action2obs_offset, d.action[0] + action2obs_offset,
                          np.cos(yaw), np.sin(yaw), color='r', scale=10)
            axs[i].axis('off')
            axs[i].title.set_text('reward ' + str(d.reward))
    fig.tight_layout()
    plt.show()
