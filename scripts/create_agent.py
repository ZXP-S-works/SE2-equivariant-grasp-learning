from agents.agents_3d.dqn_3d_asr import DQN3DASR
from utils.parameters import *
from networks.equivariant_models_refactor import EquResUReg, EquShiftQ2ResN


def createAgent():
    """
    Creates the DQN agent with equivariant q1, q2 networks
    """

    if half_rotation:
        rz_range = (0, (num_rotations - 1) * np.pi / num_rotations)
    else:
        rz_range = (0, (num_rotations - 1) * 2 * np.pi / num_rotations)

    diag_length = float(heightmap_size) * np.sqrt(2)
    diag_length = int(np.ceil(diag_length / 32) * 32)
    initialize = initialize_net
    if load_sub is not None or load_model_pre is not None:
        initialize = False

    assert args.use_depth + args.use_rgb > 0
    q1_input_channel = 1 * args.use_depth + 3 * args.use_rgb
    q2_patch_channel = q1_input_channel
    patch_shape = (q2_patch_channel, patch_size, patch_size)
    q2_input_shape = (q2_patch_channel + 1, patch_size, patch_size)
    is_fcn_si = False
    fcn_out = num_primitives

    q1 = EquResUReg(q1_input_channel, fcn_out, domain_shape=(1, diag_length, diag_length),
                    patch_shape=patch_shape, N=equi_n, flip=True, initialize=initialize, is_fcn_si=is_fcn_si,
                    last_activation_softmax=True).to(device)
    q2 = EquShiftQ2ResN(q2_input_shape, num_rotations, num_primitives, kernel_size=7, quotient=False,
                        last_quotient=True, initialize=initialize).to(device)
    agent = DQN3DASR(workspace, heightmap_size, device, lr=lr, num_primitives=num_primitives,
                     patch_size=patch_size,
                     num_rz=num_rotations, rz_range=rz_range, network=model)
    agent.initNetwork(q1, q2)
    agent.detach_es = detach_es
    agent.aug = aug
    return agent
