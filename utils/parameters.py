import numpy as np
import torch
import argparse


def strToBool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

# Network
# parser.add_argument('--network', type=str, default='ours_method',
#                     help='Network name in inference/models',
#                     choices=['ours_method', 'grconvnet', 'grconvnet2', 'grconvnet3', 'grconvnet4',
#                              'ggcnn', 'vpg', 'fcgqcnn'])
parser.add_argument('--input-size', type=int, default=224,
                    help='Input image size for the network')
parser.add_argument('--use-depth', type=int, default=1,
                    help='Use Depth image for training (1/0)')
parser.add_argument('--use-rgb', type=int, default=0,
                    help='Use RGB image for training (1/0)')
parser.add_argument('--use-dropout', type=int, default=1,
                    help='Use dropout for training (1/0)')
parser.add_argument('--dropout-prob', type=float, default=0.1,
                    help='Dropout prob for training (0-1)')
parser.add_argument('--channel-size', type=int, default=32,
                    help='Internal channel size for the network')
parser.add_argument('--iou-threshold', type=float, default=0.25,
                    help='Threshold for IOU matching')

# Datasets
parser.add_argument('--dataset', type=str, default='cornell',
                    help='Dataset Name ("cornell" or "jacquard")')
parser.add_argument('--dataset-path', type=str,
                    help='Path to dataset')
# parser.add_argument('--split', type=float, default=0.9,
#                     help='Fraction of data for training (remainder is validation)')
# parser.add_argument('--use_length', type=int, default=None,
#                     help='Dataset length')
parser.add_argument('--train_size', type=int, default=100,
                    help='Dataset length')
parser.add_argument('--test_size', type=int, default=400,
                    help='Dataset length')
parser.add_argument('--ds-shuffle', type=strToBool, default=True,
                    help='Shuffle the dataset')
parser.add_argument('--ds-rotate', type=float, default=0.0,
                    help='Shift the start point of the dataset to use a different test/train split')
parser.add_argument('--num-workers', type=int, default=2,
                    help='Dataset workers')

# Training
parser.add_argument('--batch-size', type=int, default=8,
                    help='Batch size')
# parser.add_argument('--epochs', type=int, default=50,
#                     help='Training epochs')
parser.add_argument('--train_with_centers', type=strToBool, default=False)
parser.add_argument('--train_with_y_pos', type=strToBool, default=True)
parser.add_argument('--q1_train_q2', type=int, default=10,
                    help='sampling q1_train_q2 point form q1 to train q2')
parser.add_argument('--normalize_depth', type=strToBool, default=False)
# parser.add_argument('--train_tau', type=float, default=0.01,
#                     help='training Boltzmann temperature')
# parser.add_argument('--test_tau', type=float, default=0.002,
#                     help='test Boltzmann temperature')
parser.add_argument('--batches-per-epoch', type=int, default=1000,
                    help='Batches per Epoch')
parser.add_argument('--optim', type=str, default='adam',
                    help='Optmizer for the training. (adam or SGD)')

# Logging etc.
parser.add_argument('--description', type=str, default='',
                    help='Training description')
parser.add_argument('--logdir', type=str, default='logs/',
                    help='Log directory')
parser.add_argument('--vis', action='store_true',
                    help='Visualise the training process')
parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                    help='Force code to run in CPU mode')
parser.add_argument('--random-seed', type=int, default=100,
                    help='Random seed for numpy')

env_group = parser.add_argument_group('environment')
env_group.add_argument('--env', type=str, default='random_household_picking_clutter_full_obs_30',
                       help='random_household_picking_clutter_full_obs_30')
env_group.add_argument('--reward_type', type=str, default='dense')
env_group.add_argument('--simulator', type=str, default='pybullet')
env_group.add_argument('--robot', type=str, default='kuka')
env_group.add_argument('--num_objects', type=int, default=15)
env_group.add_argument('--max_episode_steps', type=int, default=30)
env_group.add_argument('--fast_mode', type=strToBool, default=True)
env_group.add_argument('--simulate_grasp', type=strToBool, default=True)
env_group.add_argument('--action_sequence', type=str, default='xyrp')
env_group.add_argument('--random_orientation', type=strToBool, default=True)
env_group.add_argument('--num_processes', type=int, default=1)
env_group.add_argument('--render', type=strToBool, default=False)
env_group.add_argument('--workspace_size', type=float, default=0.3)
env_group.add_argument('--heightmap_size', type=int, default=128)
env_group.add_argument('--action_pixel_range', type=int, default=96)
env_group.add_argument('--action_mask', type=str, default='square', choices=['square', 'none', '? for X arm'])  #ToDO
env_group.add_argument('--patch_size', type=int, default=32)
env_group.add_argument('--perfect_grasp', action='store_true')
env_group.add_argument('--perfect_place', action='store_true')
env_group.add_argument('--in_hand_mode', type=str, default='raw', choices=['raw', 'proj'])
env_group.add_argument('--z_heuristic', type=str, default='residual',
                       choices=['patch_all', 'patch_center', 'patch_rectangular', 'residual'])

training_group = parser.add_argument_group('training')
training_group.add_argument('--alg', default='dqn_asr')
training_group.add_argument('--is_test', type=strToBool, default=False)
training_group.add_argument('--is_baseline', type=strToBool, default=False)
training_group.add_argument('--initialize_net', type=strToBool, default=True)
training_group.add_argument('--q2_train_q1', type=str, default='Boltzmann10')
training_group.add_argument('--init_curiosity_l2', type=float, default=0.0)
training_group.add_argument('--final_curiosity_l2', type=float, default=0.0)
training_group.add_argument('--td_err_measurement', default='smooth_l1',
                            choices=['smooth_l1', 'BCE', 'q1_smooth_l1_q2_BCE'])
training_group.add_argument('--q1_success_td_target', default='rewards', choices=['rewards', 'q2'])
training_group.add_argument('--q1_failure_td_target', default='non_action_max_q2',
                            choices=['rewards', 'non_action_max_q2'])
training_group.add_argument('--is_bandit', type=strToBool, default=True)
training_group.add_argument('--model', type=str, default='equ_resu_nodf_flip_softmax')
training_group.add_argument('--model_predict_width', type=strToBool, default=False)
training_group.add_argument('--num_rotations', type=int, default=8)
training_group.add_argument('--half_rotation', type=strToBool, default=True)
training_group.add_argument('--lr', type=float, default=1e-4)
training_group.add_argument('--explore', type=int, default=500)
training_group.add_argument('--step_eps', type=int, default=20)
training_group.add_argument('--z_heuristic_step', type=int, default=0)
training_group.add_argument('--init_eps', type=float, default=1.)
training_group.add_argument('--final_eps', type=float, default=0.)
training_group.add_argument('--train_tau', type=float, default=0.01)
training_group.add_argument('--test_tau', type=float, default=0.002)
training_group.add_argument('--training_iters', type=int, default=1)
training_group.add_argument('--training_offset', type=int, default=20)
training_group.add_argument('--max_episode', type=int, default=1500)
training_group.add_argument('--device_name', type=str, default='cuda')
training_group.add_argument('--use_target_net', type=strToBool, default=False)
training_group.add_argument('--target_update_freq', type=int, default=20)
training_group.add_argument('--num_saves', type=int, default=10)
training_group.add_argument('--learning_curve_avg_window', type=int, default=50)
training_group.add_argument('--action_selection', type=str, default='Boltzmann')
training_group.add_argument('--dilation_aperture', type=int, default=4)
training_group.add_argument('--q1_random_sample', type=strToBool, default=False)
training_group.add_argument('--hm_threshold', type=float, default=0.005)
training_group.add_argument('--rand_argmax_action_top_n', type=int, default=1)
training_group.add_argument('--load_model_pre', type=str, default=None)
training_group.add_argument('--sl', action='store_true')
training_group.add_argument('--note', type=str, default=None)
training_group.add_argument('--seed', type=int, default=None)
training_group.add_argument('--perlin', type=float, default=0.0)
training_group.add_argument('--onpolicy_data_D4_aug', type=strToBool, default=False)
training_group.add_argument('--onpolicy_data_aug_n', type=int, default=8)
training_group.add_argument('--onpolicy_data_aug_flip', type=strToBool, default=True)
training_group.add_argument('--onpolicy_data_aug_rotate', type=strToBool, default=True)
training_group.add_argument('--num_zs', type=int, default=16)
training_group.add_argument('--min_z', type=float, default=-0.01)
training_group.add_argument('--max_z', type=float, default=-0.13)
training_group.add_argument('--patch_div', type=float, default=1.0)
training_group.add_argument('--patch_mul', type=float, default=1.0)
training_group.add_argument('--q2_model', type=str, default='equ_shift_reg_7_lq_softmax_resnet64')
training_group.add_argument('--q3_model', type=str, default='cnn_like_equ_lq_softmax_resnet64')
training_group.add_argument('--q2_predict_width', type=strToBool, default=False)
training_group.add_argument('--q2_input', default='hm_minus_z', choices=['hm', 'hm_minus_z', 'hm_and_z'])
training_group.add_argument('--q3_input', default='hm_minus_z', choices=['hm', 'hm_minus_z', 'hm_and_z'])
training_group.add_argument('--q3_adjustment', default='none', choices=['none', 'add_bias', 'uniform_random'])
training_group.add_argument('--equi_n', type=int, default=4)
training_group.add_argument('--detach_es', action='store_true')
training_group.add_argument('--aug', type=int, default=0)
training_group.add_argument('--aug_continuous_theta', type=strToBool, default=False)

eval_group = parser.add_argument_group('eval')
eval_group.add_argument('--eval_num_processes', default=20, type=int)
eval_group.add_argument('--num_eval', default=10, type=int)
eval_group.add_argument('--num_eval_episodes', default=1000, type=int)

margin_group = parser.add_argument_group('margin')
margin_group.add_argument('--margin_l', type=float, default=0.1)
margin_group.add_argument('--margin_weight', type=float, default=0.1)

buffer_group = parser.add_argument_group('buffer')
buffer_group.add_argument('--buffer', default='expert', choices=['normal', 'per', 'expert', 'per_expert'])
buffer_group.add_argument('--sample_onpolicydata', type=strToBool, default=True)
buffer_group.add_argument('--onlyfailure', type=int, default=4)
buffer_group.add_argument('--success_to_expert', type=strToBool, default=False)
buffer_group.add_argument('--per_eps', type=float, default=1e-6, help='Epsilon parameter for PER')
buffer_group.add_argument('--per_alpha', type=float, default=0.6, help='Alpha parameter for PER')
buffer_group.add_argument('--per_beta', type=float, default=0.4, help='Initial beta parameter for PER')
buffer_group.add_argument('--per_expert_eps', type=float, default=1)
buffer_group.add_argument('--per_td_error', type=str, default='last',
                          choices=['all', 'last', 'last_square', 'last_BCE'])
buffer_group.add_argument('--batch_size', type=int, default=8)
buffer_group.add_argument('--buffer_size', type=int, default=100000)
buffer_group.add_argument('--fixed_buffer', action='store_true')

logging_group = parser.add_argument_group('logging')
logging_group.add_argument('--log_pre', type=str, default='/tmp')
logging_group.add_argument('--log_sub', type=str, default=None)
logging_group.add_argument('--no_bar', action='store_true')
logging_group.add_argument('--time_limit', type=float, default=10000)
logging_group.add_argument('--load_sub', type=str, default=None)

test_group = parser.add_argument_group('test')
# test_group.add_argument('--test', action='store_true')
args = parser.parse_args()
# env
random_orientation = args.random_orientation
reward_type = args.reward_type
env = args.env
simulator = args.simulator
num_objects = args.num_objects
max_episode_steps = args.max_episode_steps
fast_mode = args.fast_mode
simulate_grasp = args.simulate_grasp
action_sequence = args.action_sequence
num_processes = args.num_processes
render = args.render
perfect_grasp = args.perfect_grasp
perfect_place = args.perfect_place
scale = 1.
robot = args.robot
heightmap_size = args.heightmap_size
action_mask = args.action_mask
action_pixel_range = args.action_pixel_range
patch_size = args.patch_size

workspace_size = args.workspace_size
if env == 'DualBinFrontRear':
    # action_sequence = 'xyrp'
    # robot_ws_center = [-0.3426, 0.048, -0.114]  # aggressively not safe
    robot_ws_center = [-0.3426, 0.048, -0.112]  # aggressively not safe
    # robot_ws_center = [-0.3426, 0.048, -0.110]  # broader not safe
    # robot_ws_center = [-0.3426, 0.048, -0.106]  # broader not safe
    # robot_ws_center = [-0.3426, 0.048, -0.085]  # broader safe
    pick_place_offset = 0.12
    cam_size = 256
    real_workspace_size = 0.8
    workspace_size = 0.25
    action_space_size = 0.25
    num_rotations = 16
    obs_source = 'reconstruct'
    place_open_pos = 0
    workspace = np.asarray([[-workspace_size / 2, workspace_size / 2],
                            [-workspace_size / 2, workspace_size / 2],
                            [robot_ws_center[2], robot_ws_center[2] + 0.4]])
    real_workspace = np.asarray(
        [[robot_ws_center[0] - real_workspace_size / 2, robot_ws_center[0] + real_workspace_size / 2],
         [robot_ws_center[1] - real_workspace_size / 2, robot_ws_center[1] + real_workspace_size / 2],
         [robot_ws_center[2], robot_ws_center[2] + 0.4]])
    # render = True
    safe_z_region = 1 / 4
    pixel_size = real_workspace_size / cam_size
elif env == 'random_household_picking_clutter_full_obs_30':
    env = 'random_household_picking_clutter_full_obs'
    workspace_size = 0.4
    heightmap_size = 128
    action_space_size = 0.3
    action_pixel_range = 96
    workspace = np.asarray([[0.5 - workspace_size / 2, 0.5 + workspace_size / 2],
                            [0 - workspace_size / 2, 0 + workspace_size / 2],
                            [0, 0 + workspace_size]])
    safe_z_region = 1 / 4
    pixel_size = 0.4 / 128
else:
    action_space_size = workspace_size
    workspace = np.asarray([[0.5 - workspace_size / 2, 0.5 + workspace_size / 2],
                            [0 - workspace_size / 2, 0 + workspace_size / 2],
                            [0, 0 + workspace_size]])


num_primitives = 2

heightmap_resolution = workspace_size / heightmap_size
action_space = [0, heightmap_size]

num_rotations = args.num_rotations
half_rotation = args.half_rotation
if half_rotation:
    rotations = [np.pi / num_rotations * i for i in range(num_rotations)]
else:
    rotations = [(2 * np.pi) / num_rotations * i for i in range(num_rotations)]
in_hand_mode = args.in_hand_mode
z_heuristic = args.z_heuristic

######################################################################################
# training
alg = args.alg
is_test = args.is_test
is_baseline = args.is_baseline
initialize_net = args.initialize_net
q2_train_q1 = args.q2_train_q1
init_curiosity_l2 = args.init_curiosity_l2
final_curiosity_l2 = args.final_curiosity_l2
if alg == 'dqn_sl_anneal':
    args.sl = True
td_err_measurement = args.td_err_measurement
q1_success_td_target = args.q1_success_td_target
q1_failure_td_target = args.q1_failure_td_target
is_bandit = args.is_bandit
model = args.model
model_predict_width = args.model_predict_width
lr = args.lr
explore = args.explore
step_eps = args.step_eps
z_heuristic_step = args.z_heuristic_step
init_eps = args.init_eps
final_eps = args.final_eps
train_tau = args.train_tau
test_tau = args.test_tau
training_iters = args.training_iters
training_offset = args.training_offset
max_episode = args.max_episode
device = torch.device(args.device_name)
use_target_net = args.use_target_net
target_update_freq = args.target_update_freq
num_saves = args.num_saves
learning_curve_avg_window = args.learning_curve_avg_window
action_selection = args.action_selection
dilation_aperture = args.dilation_aperture
q1_random_sample = args.q1_random_sample
hm_threshold = args.hm_threshold
sl = args.sl

load_model_pre = args.load_model_pre
note = args.note
seed = args.seed
perlin = args.perlin
onpolicy_data_D4_aug = args.onpolicy_data_D4_aug
if onpolicy_data_D4_aug:
    aug_scale = 8
else:
    aug_scale = 1

onpolicy_data_aug_n = args.onpolicy_data_aug_n
onpolicy_data_aug_flip = args.onpolicy_data_aug_flip
onpolicy_data_aug_rotate = args.onpolicy_data_aug_rotate
rand_argmax_action_top_n = args.rand_argmax_action_top_n
patch_div = args.patch_div
patch_mul = args.patch_mul

q2_model = args.q2_model
q2_input = args.q2_input
q3_model = args.q3_model
q3_input = args.q3_input
q3_adjustment = args.q3_adjustment
equi_n = args.equi_n

detach_es = args.detach_es
aug = args.aug
aug_continuous_theta = args.aug_continuous_theta

# eval
eval_num_processes = args.eval_num_processes
num_eval = args.num_eval
eval_freq = int(max_episode / num_eval)
num_eval_episodes = args.num_eval_episodes

# buffer
buffer_type = args.buffer
sample_onpolicydata = args.sample_onpolicydata
onlyfailure = args.onlyfailure
success_to_expert = args.success_to_expert
per_eps = args.per_eps
per_alpha = args.per_alpha
per_beta = args.per_beta
per_expert_eps = args.per_expert_eps
per_td_error = args.per_td_error
sample_batch_size = args.batch_size
if aug > 0:
    batch_size = aug * sample_batch_size
else:
    batch_size = sample_batch_size
buffer_size = args.buffer_size
fixed_buffer = args.fixed_buffer

# margin
margin_l = args.margin_l
margin_weight = args.margin_weight

# logging
log_pre = args.log_pre
log_sub = args.log_sub
no_bar = args.no_bar
time_limit = args.time_limit
load_sub = args.load_sub
if load_sub == 'None':
    load_sub = None

# z
num_zs = args.num_zs
min_z = args.min_z
max_z = args.max_z

######################################################################################
env_config = {'workspace': workspace, 'max_steps': max_episode_steps, 'obs_size': heightmap_size,
              'in_hand_size': patch_size, 'bin_size': action_space_size, 'fast_mode': fast_mode,
              'action_sequence': action_sequence, 'render': render, 'num_objects': num_objects,
              'random_orientation': random_orientation, 'reward_type': reward_type, 'simulate_grasp': simulate_grasp,
              'perfect_grasp': perfect_grasp, 'perfect_place': perfect_place, 'scale': scale, 'robot': robot,
              'workspace_check': 'point', 'in_hand_mode': in_hand_mode, 'object_scale_range': (0.6, 0.6),
              'hard_reset_freq': 1000, 'physics_mode': 'fast', 'z_heuristic': z_heuristic}
planner_config = {'pos_noise': 0., 'rot_noise': 0.,
                  'random_orientation': random_orientation, 'half_rotation': half_rotation}

if env in ['random_block_picking_clutter']:
    env_config['object_scale_range'] = (0.8, 0.8)
    env_config['min_object_distance'] = 0
    env_config['min_boarder_padding'] = 0.15
    env_config['adjust_gripper_after_lift'] = True
if env in ['DualBinFrontRear']:
    env_config['object_scale_range'] = (1, 1)
    env_config['reward_type'] = 'dense'
if env in ['random_household_picking_clutter',
           'random_household_picking_individual',
           'random_household_picking_clutter_full_obs',
           'random_household_picking_clutter_full_obs_30']:
    env_config['object_scale_range'] = (1, 1)
    env_config['min_object_distance'] = 0
    env_config['min_boarder_padding'] = 0.2
    env_config['reward_type'] = 'dense'
    env_config['adjust_gripper_after_lift'] = True
if seed is not None:
    env_config['seed'] = seed
######################################################################################
hyper_parameters = {}
for key in sorted(vars(args)):
    hyper_parameters[key] = vars(args)[key]

for key in hyper_parameters:
    print('{}: {}'.format(key, hyper_parameters[key]))


def parse_args():
    return args
