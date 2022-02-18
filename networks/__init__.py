def get_network(network_name):
    network_name = network_name.lower()
    # ours
    if network_name in ['ours_method', 'equ_resu_nodf_flip_softmax']:
        from .ours_method_SL_wrapper import OursMethod
        return OursMethod
    # GR-Conv-Net and GG-CNN
    elif network_name in ['grconvnet', 'ggcnn']:
        from .grconvnet_ggcnn_SL_wrapper import GRCONVNETGGCNN
        return GRCONVNETGGCNN
    # VPG and FC-GQ-CNN
    elif network_name in ['vpg', 'fcgqcnn']:
        from .vpg_fcgqcnn_SL_wrapper import VPGFCGQCNN
        return VPGFCGQCNN
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
