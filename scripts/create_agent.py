from agents.agents_3d.dqn_3d_asr_drq import DQN3DASRDrQ
from agents.agents_3d.dqn_3d_fcn import DQN3DFCN
from agents.agents_3d.dqn_3d_fcn_si import DQN3DFCNSingleIn
from agents.agents_3d.dqn_3d_asr import DQN3DASR
from agents.agents_3d.dqn_3d_asr_deictic import DQN3DASRDeictic
from agents.agents_4d.dqn_4d_asr import DQN4DASR
from agents.agents_4d.dqn_4d_asr_deictic import DQN4DASRDeictic
from networks.ggcnn import GGCNN
from networks.grconvnet import GenerativeResnet

from utils.parameters import *
from networks.models import CNN, ResUSoftmax, ResU_like_EquResUReg, CNN_like_EquResQ2
from networks.vpg import VisualPushingGrasping
from networks.fc_gq_cnn import FCGQCNN
from networks.equivariant_models_refactor import EquResUReg, EquShiftQ23, EquResU2MReg, EquResU2MLReg, EquShiftQ2ResN


def createAgent():
    if half_rotation:
        rz_range = (0, (num_rotations - 1) * np.pi / num_rotations)
    else:
        rz_range = (0, (num_rotations - 1) * 2 * np.pi / num_rotations)

    diag_length = float(heightmap_size) * np.sqrt(2)
    diag_length = int(np.ceil(diag_length / 32) * 32)

    assert args.use_depth + args.use_rgb > 0
    q1_input_channel = 1 * args.use_depth + 3 * args.use_rgb
    q2_patch_channel = q1_input_channel
    if q2_input == 'hm_and_z':
        q2_patch_channel += 1

    if q3_input in ['hm', 'hm_minus_z'] or alg.find('deictic') != -1:
        q3_patch_channel = 1
    elif q3_input == 'hm_and_z':
        q3_patch_channel = 2

    patch_shape = (q2_patch_channel, patch_size, patch_size)

    fcn_out = num_primitives
    is_fcn_si = False
    if alg.find('fcn_si') > -1:
        if model.find('equ') > -1:
            is_fcn_si = True
        else:
            fcn_out = num_rotations * num_primitives

    initialize = initialize_net

    if load_sub is not None or load_model_pre is not None:
        initialize = False

    ###################################################################################
    if model == 'resu':
        fcn = ResUSoftmax(q1_input_channel, fcn_out, domain_shape=(1, diag_length, diag_length),
                          patch_shape=patch_shape, last_activation_softmax=False).to(device)
    elif model == 'resu_softmax':
        fcn = ResUSoftmax(q1_input_channel, fcn_out, domain_shape=(1, diag_length, diag_length),
                          patch_shape=patch_shape).to(device)
    elif model == 'resu_like_equ_resu':
        fcn = ResU_like_EquResUReg(q1_input_channel, fcn_out, domain_shape=(1, diag_length, diag_length),
                                   N=equi_n).to(device)
    elif model == 'grconvnet':
        fcn = GenerativeResnet(q1_input_channel).to(device)
    elif model == 'ggcnn':
        fcn = GGCNN(q1_input_channel).to(device)
    elif model == 'vpg':
        fcn = VisualPushingGrasping(q1_input_channel, predict_width=model_predict_width).to(device)
    elif model == 'fcgqcnn':
        fcn = FCGQCNN(q1_input_channel, fcn_out, domain_shape=(1, diag_length, diag_length),
                      patch_shape=patch_shape, predict_width=model_predict_width).to(device)
    elif model == 'equ_resu_nodf':
        fcn = EquResUReg(q1_input_channel, fcn_out, domain_shape=(1, diag_length, diag_length),
                         patch_shape=patch_shape, N=equi_n, initialize=initialize, is_fcn_si=is_fcn_si).to(device)
    elif model == 'equ_resu_nodf_softmax':
        fcn = EquResUReg(q1_input_channel, fcn_out, domain_shape=(1, diag_length, diag_length),
                         patch_shape=patch_shape, N=equi_n, initialize=initialize, is_fcn_si=is_fcn_si,
                         last_activation_softmax=True).to(device)
    elif model == 'equ_resu_nodf_flip':
        fcn = EquResUReg(q1_input_channel, fcn_out, domain_shape=(1, diag_length, diag_length),
                         patch_shape=patch_shape, N=equi_n, flip=True, initialize=initialize,
                         is_fcn_si=is_fcn_si).to(device)
    elif model == 'equ_resu_nodf_flip_softmax':
        fcn = EquResUReg(q1_input_channel, fcn_out, domain_shape=(1, diag_length, diag_length),
                         patch_shape=patch_shape, N=equi_n, flip=True, initialize=initialize, is_fcn_si=is_fcn_si,
                         last_activation_softmax=True).to(device)
    elif model == 'equ_resu2m_nodf_flip_softmax':
        fcn = EquResU2MReg(q1_input_channel, fcn_out, domain_shape=(1, diag_length, diag_length),
                           patch_shape=patch_shape, N=equi_n, flip=True, initialize=initialize, is_fcn_si=is_fcn_si,
                           last_activation_softmax=True).to(device)
    elif model == 'equ_resu2m_large_nodf_flip_softmax':
        fcn = EquResU2MLReg(q1_input_channel, fcn_out, domain_shape=(1, diag_length, diag_length),
                            patch_shape=patch_shape, N=equi_n, flip=True, n_middle_channels=(32, 64, 128, 128),
                            initialize=initialize, is_fcn_si=is_fcn_si, last_activation_softmax=True).to(device)


    ####################################################

    if alg.find('asr') > -1:
        if alg.find('deictic') > -1:
            q2_output_size = num_primitives
        else:
            q2_output_size = num_primitives * num_rotations
        q2_input_shape = (q2_patch_channel + 1, patch_size, patch_size)

        ###################################################################################
        if q2_model == 'cnn_no_sharing':
            q2 = CNN(q2_input_shape, q2_output_size, last_activation_softmax=False).to(device)
        elif q2_model == 'cnn_no_sharing_softmax':
            q2 = CNN(q2_input_shape, q2_output_size, last_activation_softmax=True).to(device)
        elif q2_model == 'cnn_like_equ_lq_softmax_resnet64':
            q2 = CNN_like_EquResQ2(q2_input_shape, num_rotations, num_primitives).to(device)
        elif q2_model == 'equ_shift_reg_7_lq':
            q2 = EquShiftQ23(q2_input_shape, num_rotations, num_primitives, kernel_size=7, n_hidden=32, quotient=False,
                             last_quotient=True, initialize=initialize, last_activation_softmax=False).to(device)
        elif q2_model == 'equ_shift_reg_7_lq_softmax':
            q2 = EquShiftQ23(q2_input_shape, num_rotations, num_primitives, kernel_size=7, n_hidden=32, quotient=False,
                             last_quotient=True, initialize=initialize).to(device)
        elif q2_model == 'equ_shift_reg_7_lq_softmax_resnet32':
            q2 = EquShiftQ2ResN(q2_input_shape, num_rotations, num_primitives, kernel_size=7, n_hidden=32,
                                quotient=False, last_quotient=True, initialize=initialize).to(device)
        elif q2_model == 'equ_shift_reg_7_lq_resnet64':
            q2 = EquShiftQ2ResN(q2_input_shape, num_rotations, num_primitives, kernel_size=7, quotient=False,
                                last_quotient=True, initialize=initialize, last_activation_softmax=False).to(device)
        elif q2_model == 'equ_shift_reg_7_lq_softmax_resnet64':
            q2 = EquShiftQ2ResN(q2_input_shape, num_rotations, num_primitives, kernel_size=7, quotient=False,
                                last_quotient=True, initialize=initialize).to(device)
        elif q2_model == 'equ_shift_reg_7_lq_last_no_maxpool32':
            q2 = EquShiftQ23(q2_input_shape, num_rotations, num_primitives, kernel_size=7, n_hidden=32,
                             quotient=False, last_quotient=True, initialize=initialize,
                             last_activation_softmax=False, q2_type='convolution_last_no_maxpool').to(device)
        elif q2_model == 'equ_shift_reg_7_lq_softmax_last_no_maxpool32':
            q2 = EquShiftQ23(q2_input_shape, num_rotations, num_primitives, kernel_size=7, n_hidden=32, quotient=False,
                             last_quotient=True, initialize=initialize,
                             q2_type='convolution_last_no_maxpool').to(device)
        elif q2_model in ['None', 'none']:
            q2 = None
        else:
            raise NotImplementedError

        ######################################################################333
        if action_sequence == 'xyzrp':
            q3_input_shape = (q3_patch_channel + 1, patch_size, patch_size)
            if alg.find('deictic') > -1:
                q3_output_size = num_primitives
            else:
                q3_output_size = num_primitives * num_zs
            if q3_model == 'cnn_no_sharing':
                q3 = CNN(q3_input_shape, q3_output_size, last_activation_softmax=False).to(device)
            elif q3_model == 'cnn_no_sharing_softmax':
                q3 = CNN(q3_input_shape, q3_output_size, last_activation_softmax=True).to(device)
            elif q3_model == 'cnn_like_equ_lq_softmax_resnet64':
                q3 = CNN_like_EquResQ2(q3_input_shape, q3_output_size // num_primitives, num_primitives).to(device)
            elif q3_model in ['None', 'none']:
                q3 = None
            else:
                raise NotImplementedError

    if action_sequence == 'xyrp':
        if alg.find('asr') > -1:
            if alg == 'dqn_asr':
                agent = DQN3DASR(workspace, heightmap_size, device, lr=lr, sl=sl, num_primitives=num_primitives,
                                 patch_size=patch_size,
                                 num_rz=num_rotations, rz_range=rz_range, network=model)
                agent.initNetwork(fcn, q2)
            elif alg == 'dqn_asr_drq':
                agent = DQN3DASRDrQ(workspace, heightmap_size, device, lr=lr, sl=sl, num_primitives=num_primitives,
                                    patch_size=patch_size,
                                    num_rz=num_rotations, rz_range=rz_range)
                agent.initNetwork(fcn, q2)
            elif alg == 'dqn_asr_deictic':
                agent = DQN3DASRDeictic(workspace, heightmap_size, device, lr=lr, sl=sl, num_primitives=num_primitives,
                                        patch_size=patch_size,
                                        num_rz=num_rotations, rz_range=rz_range)
                agent.initNetwork(fcn, q2)
            else:
                raise NotImplementedError

        elif alg.find('fcn_si') > -1:
            if alg == 'dqn_fcn_si':
                agent = DQN3DFCNSingleIn(workspace, heightmap_size, device, lr=lr, sl=sl, num_primitives=num_primitives,
                                         patch_size=patch_size,
                                         num_rz=num_rotations, rz_range=rz_range)
            else:
                raise NotImplementedError
            agent.initNetwork(fcn)

        elif alg.find('fcn') > -1:
            if alg == 'dqn_fcn':
                agent = DQN3DFCN(workspace, heightmap_size, device, lr=lr, sl=sl, num_primitives=num_primitives,
                                 patch_size=patch_size,
                                 num_rz=num_rotations, rz_range=rz_range)
            else:
                raise NotImplementedError
            agent.initNetwork(fcn)

    elif action_sequence == 'xyzrp':
        if alg.find('asr') > -1:
            if alg == 'dqn_asr':
                agent = DQN4DASR(workspace, heightmap_size, device, lr=lr, sl=sl, num_primitives=num_primitives,
                                 patch_size=patch_size, rz_range=rz_range, num_rz=num_rotations, num_zs=num_zs,
                                 z_range=(min_z, max_z))
            elif alg == 'dqn_asr_deictic':
                agent = DQN4DASRDeictic(workspace, heightmap_size, device, lr=lr, sl=sl, num_primitives=num_primitives,
                                        patch_size=patch_size, rz_range=rz_range, num_rz=num_rotations,
                                        num_zs=num_zs, z_range=(min_z, max_z))
            else:
                raise NotImplementedError

            agent.initNetwork(fcn, q2, q3)

    agent.detach_es = detach_es
    agent.per_td_error = per_td_error
    agent.aug = aug
    return agent
