import logging
import os
import sys
import time
sys.path.append('./')
sys.path.append('..')
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
if '/home/zxp-s-works/robotic-grasping' in sys.path:
    sys.path.remove('/home/zxp-s-works/robotic-grasping')

import cv2
import torch.optim as optim
import torch.utils.data

from hardware.device import get_device
from networks import get_network
from inference.post_process import post_process_output
from sl_utils.data import get_dataset
from sl_utils.dataset_processing import evaluation
from sl_utils.visualisation.gridshow import gridshow
from utils.parameters import *
from utils.logger import Logger
from scripts.create_agent import createAgent


def saveModelAndInfo(logger, agent=None):
    logger.saveModel(logger.num_steps, env, agent)
    logger.saveLossCurve(1)
    # logger.saveTdErrorCurve(100)
    logger.saveRewards()
    logger.saveLosses()
    # logger.saveLearningCurve(learning_curve_avg_window)
    logger.saveEvalCurve()
    logger.saveEvalRewards()


def validate(net, device, val_data, iou_threshold):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :return: Successes, Failures and Losses
    """
    net.eval()
    val_data.dataset.is_training = False

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        for x, y, didx, rot, zoom_factor in val_data:
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            s = evaluation.calculate_iou_match(q_out,
                                               ang_out,
                                               val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                               no_grasps=1,
                                               grasp_width=w_out,
                                               threshold=iou_threshold)

            if render:
                evaluation.visualize_grasps(x, q_out, ang_out, val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                            no_grasps=1, grasp_width=w_out)

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()
    train_data.dataset.is_training = True

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx <= batches_per_epoch:
        for x, y, _, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 20 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                      lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
                          (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    start_time = time.time()
    args = parse_args()
    envs = None

    # setup agent; load the network
    logging.info('Loading Network...')
    # Get the compute device
    device = get_device(args.force_cpu)
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    network = get_network(args.model)
    agent = createAgent()
    agent.train()
    if load_model_pre:
        agent.loadModel(load_model_pre)
    net = network(agent, args).to(device)

    # if args.model in ['ours_method', 'vpg', 'fcgqcnn']:
    #     agent = createAgent()
    #     net = network(agent, args)
    # elif args.model == 'ggcnn':
    #     net = network(input_channels=input_channels)
    # else:
    #     net = network(
    #         input_channels=input_channels,
    #         dropout=args.use_dropout,
    #         prob=args.dropout_prob,
    #         channel_size=args.channel_size
    #     )
    logging.info('Done')

    if args.model in ['ours_method', 'equ_resu_nodf_flip_softmax', 'fcgqcnn']:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.optim.lower() == 'adam' or args.model == 'ggcnn':
        optimizer = optim.Adam(net.parameters())
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    elif model == 'vpg':
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    # logging
    log_dir = os.path.join(log_pre, '{}_{}_{}_{}'
                           .format(args.model, 'SL', args.dataset.title(), args.train_size))
    if note:
        log_dir += '_'
        log_dir += note

    logger = Logger(log_dir, env, 'train', num_processes, max_episode, log_sub)
    hyper_parameters['model_shape'] = agent.getModelStr()
    logger.saveParameters(hyper_parameters)

    replay_buffer = None

    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), envs, agent, replay_buffer)

    logging.root.handlers = []

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      random_rotate=True,
                      random_zoom=False,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb)
    logging.info('Dataset size is {}'.format(dataset.length))

    # Creating data indices for training and validation splits
    if args.train_size is not None:
        train_len = args.train_size
    else:
        train_len = dataset.length
    val_len = args.test_size
    indices = list(range(dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)

    # We choose the first val_len data to be validation data
    # so that validation data is fixed no matter what train_len is
    val_indices, train_indices = indices[:val_len], indices[val_len:val_len+train_len]
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')

    # Print model architecture.
    # summary(net, (input_channels, args.input_size, args.input_size))
    # f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    # sys.stdout = f
    # summary(net, (input_channels, args.input_size, args.input_size))
    # sys.stdout = sys.__stdout__
    # f.close()

    # best_iou = 0.0
    while logger.num_episodes < max_episode:
        logging.info('Beginning Epoch {:02d}'.format(logger.num_episodes))
        epoch_start = time.time()
        train_results = train(logger.num_episodes, net, device, train_data,
                              optimizer, args.batches_per_epoch, vis=args.vis)

        # # Log training losses to tensorboard
        # tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        # for n, l in train_results['losses'].items():
        #     tb.add_scalar('train_loss/' + n, l, epoch)

        # Log training losses
        logger.trainingBookkeeping(train_results['loss'], 0)
        logger.num_training_steps += 1
        logger.num_episodes += 1

        # Run Validation
        valid_start = time.time()
        epoch_t_cost = (valid_start - epoch_start) / eval_freq
        logger.SGD_time.append(epoch_t_cost)  # SGD time equal to a training epoch time
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.iou_threshold)
        eval_t_cost = time.time() - valid_start
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        logging.info('train t %f, eval t %f' % (epoch_t_cost, eval_t_cost))
        # # Log validation results to tensorbaord
        # tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        # tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        # for n, l in test_results['losses'].items():
        #     tb.add_scalar('val_loss/' + n, l, epoch)

        # Log validation results
        logger.eval_rewards.append(test_results['correct'] / (test_results['correct'] + test_results['failed']))

        # Time limit
        if (time.time() - start_time) / 3600 > time_limit:
            break

        # Save best performing network
        # iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        # if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
        #     torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
        #     best_iou = iou

        # save info
        if (logger.num_episodes + 1) % max((max_episode // num_saves), 1) == 0:
            agent.train()
            saveModelAndInfo(logger, agent)

    agent.train()
    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(args, envs, agent, replay_buffer)


if __name__ == '__main__':
    run()
