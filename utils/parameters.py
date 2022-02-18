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

parser.add_argument('--use-depth', type=int, default=1,
                    help='Use Depth image for training (1/0)')
parser.add_argument('--use-rgb', type=int, default=0,
                    help='Use RGB image for training (1/0)')

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
env_group.add_argument('--action_mask', type=str, default='square', choices=['square'])
env_group.add_argument('--patch_size', type=int, default=32)
env_group.add_argument('--perfect_grasp', action='store_true')
env_group.add_argument('--perfect_place', action='store_true')
env_group.add_argument('--in_hand_mode', type=str, default='raw', choices=['raw', 'proj'])

training_group = parser.add_argument_group('training')
training_group.add_argument('--alg', default='dqn_asr')
training_group.add_argument('--is_test', type=strToBool, default=False)
training_group.add_argument('--is_baseline', type=strToBool, default=False)
training_group.add_argument('--initialize_net', type=strToBool, default=True)
training_group.add_argument('--q2_train_q1', type=str, default='Boltzmann10')
training_group.add_argument('--model', type=str, default='equ_resu_nodf_flip_softmax')
training_group.add_argument('--model_predict_width', type=strToBool, default=False)
training_group.add_argument('--num_rotations', type=int, default=8)
training_group.add_argument('--half_rotation', type=strToBool, default=True)
training_group.add_argument('--lr', type=float, default=1e-4)
training_group.add_argument('--step_eps', type=int, default=20)
training_group.add_argument('--init_eps', type=float, default=1.)
training_group.add_argument('--final_eps', type=float, default=0.)
training_group.add_argument('--train_tau', type=float, default=0.01)
training_group.add_argument('--test_tau', type=float, default=0.002)
training_group.add_argument('--training_iters', type=int, default=1)
training_group.add_argument('--training_offset', type=int, default=20)
training_group.add_argument('--max_episode', type=int, default=1500)
training_group.add_argument('--device_name', type=str, default='cuda')
training_group.add_argument('--num_saves', type=int, default=10)
training_group.add_argument('--learning_curve_avg_window', type=int, default=50)
training_group.add_argument('--action_selection', type=str, default='Boltzmann')
training_group.add_argument('--dilation_aperture', type=int, default=4)
training_group.add_argument('--hm_threshold', type=float, default=0.005)
training_group.add_argument('--load_model_pre', type=str, default=None)
training_group.add_argument('--note', type=str, default=None)
training_group.add_argument('--seed', type=int, default=None)
training_group.add_argument('--onpolicy_data_aug_n', type=int, default=8)
training_group.add_argument('--onpolicy_data_aug_flip', type=strToBool, default=True)
training_group.add_argument('--onpolicy_data_aug_rotate', type=strToBool, default=True)
training_group.add_argument('--num_zs', type=int, default=16)
training_group.add_argument('--min_z', type=float, default=0.01)
training_group.add_argument('--max_z', type=float, default=0.13)
training_group.add_argument('--patch_div', type=float, default=1.0)
training_group.add_argument('--patch_mul', type=float, default=1.0)
training_group.add_argument('--q2_model', type=str, default='equ_shift_reg_7_lq_softmax_resnet64')
training_group.add_argument('--q2_predict_width', type=strToBool, default=False)
training_group.add_argument('--equi_n', type=int, default=4)
training_group.add_argument('--detach_es', action='store_true')
training_group.add_argument('--aug', type=int, default=0)
training_group.add_argument('--aug_continuous_theta', type=strToBool, default=False)

eval_group = parser.add_argument_group('eval')
eval_group.add_argument('--eval_num_processes', default=10, type=int)
eval_group.add_argument('--num_eval', default=10, type=int)
eval_group.add_argument('--num_eval_episodes', default=1000, type=int)

buffer_group = parser.add_argument_group('buffer')
buffer_group.add_argument('--sample_onpolicydata', type=strToBool, default=True)
buffer_group.add_argument('--onlyfailure', type=int, default=4)
buffer_group.add_argument('--success_to_expert', type=strToBool, default=False)
buffer_group.add_argument('--batch_size', type=int, default=8)
buffer_group.add_argument('--buffer_size', type=int, default=100000)
buffer_group.add_argument('--fixed_buffer', action='store_true')

logging_group = parser.add_argument_group('logging')
logging_group.add_argument('--log_pre', type=str, default='/tmp')
logging_group.add_argument('--log_sub', type=str, default=None)
logging_group.add_argument('--no_bar', action='store_true')
logging_group.add_argument('--load_sub', type=str, default=None)

test_group = parser.add_argument_group('test')
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
    robot_ws_center = [-0.3426, 0.048, -0.112]
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

######################################################################################
# training
alg = args.alg
is_test = args.is_test
is_baseline = args.is_baseline
initialize_net = args.initialize_net
q2_train_q1 = args.q2_train_q1
model = args.model
model_predict_width = args.model_predict_width
lr = args.lr
step_eps = args.step_eps
init_eps = args.init_eps
final_eps = args.final_eps
train_tau = args.train_tau
test_tau = args.test_tau
training_iters = args.training_iters
training_offset = args.training_offset
max_episode = args.max_episode
device = torch.device(args.device_name)
num_saves = args.num_saves
learning_curve_avg_window = args.learning_curve_avg_window
action_selection = args.action_selection
dilation_aperture = args.dilation_aperture
hm_threshold = args.hm_threshold

load_model_pre = args.load_model_pre
note = args.note
seed = args.seed
aug_scale = 1

onpolicy_data_aug_n = args.onpolicy_data_aug_n
onpolicy_data_aug_flip = args.onpolicy_data_aug_flip
onpolicy_data_aug_rotate = args.onpolicy_data_aug_rotate
patch_div = args.patch_div
patch_mul = args.patch_mul

q2_model = args.q2_model
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
sample_onpolicydata = args.sample_onpolicydata
onlyfailure = args.onlyfailure
success_to_expert = args.success_to_expert
sample_batch_size = args.batch_size
if aug > 0:
    batch_size = aug * sample_batch_size
else:
    batch_size = sample_batch_size
buffer_size = args.buffer_size
fixed_buffer = args.fixed_buffer

# logging
log_pre = args.log_pre
log_sub = args.log_sub
no_bar = args.no_bar
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
              'hard_reset_freq': 1000, 'physics_mode': 'fast', 'z_heuristic': 'patch_center'}
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
