# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import matplotlib.pyplot as plt

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math

from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections

#from common.model_poseformer import *
from common.layers import *

from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import *
import pandas as pd
from scipy.spatial.distance import pdist


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# print(torch.cuda.device_count())

###################
args = parse_args()
# print(args)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

#print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

###################

for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is mibatch_3dssing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]


    return out_camera_params, out_poses_3d, out_poses_2d

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

################################################

divide_data = int(args.devide_data)
embed_dim_ratio = int(args.embed_dim_ratio)
depth = int(args.depth)
num_heads = int(args.num_heads)
drop_rate = args.drop_rate
attn_drop_rate = args.attn_drop_rate
drop_path_rate = args.drop_path_rate
mlp_ratio = int(args.mlp_ratio)
weight_decay = args.weight_decay


# =============================================================================
divide_data = 4
# embed_dim_ratio = 32
# depth = 4
# num_heads = 8
# drop_rate = 0.0
# attn_drop_rate = 0.0
# drop_path_rate = 0.0
# mlp_ratio = 2
# weight_decay = 0.0
# =============================================================================

args.batch_size = 512


print('batch_size: '+str(args.batch_size)+' checkpoint: '+str(args.checkpoint))
print('divide_data: '+str(divide_data)+' embed_dim_ratio: '+str(embed_dim_ratio)+' depth: '+str(depth)+' num_heads: '+str(num_heads)+' mlp_ratio: '+str(mlp_ratio))
print('drop_rate: '+str(drop_rate)+' attn_drop_rate: '+str(attn_drop_rate)+' drop_path_rate: '+str(drop_path_rate)+' weight_decay: '+str(weight_decay))

#####################################
args.export_training_curves = True
args.checkpoint_frequency = 5
#args.eval_de = True

receptive_field = 5
estimation_field = 20
frames_gap = 5

print('INFO: Receptive field: {} frames'.format(receptive_field))
print('INFO: Estimation field: {} frames'.format(estimation_field))
print()

pad = (receptive_field -1) // 2 # Padding on each side
min_loss = 100000
width = cam['res_w']
height = cam['res_h']
num_joints = keypoints_metadata['num_joints']

args.learning_rate = 0.00004  # default: 0.00004
args.lr_decay = 0.96
args.keypoints = 'cpn_ft_h36m_dbb'

#########################################PoseTransformer

model_pos_train = My_Transformer(num_frame=receptive_field, num_frame_d = estimation_field ,num_joints=num_joints, in_chans=3, embed_dim_ratio=embed_dim_ratio, depth=depth,
        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)

model_pos = Inference(num_frame=receptive_field, num_frame_d = estimation_field , num_joints=num_joints, in_chans=3, embed_dim_ratio=embed_dim_ratio, depth=depth,
        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,drop_path_rate=0)


################ load weight ########################
#posetrans_checkpoint = torch.load('./checkpoint20/last_epoch.bin', map_location=lambda storage, loc: storage)
#posetrans_checkpoint = posetrans_checkpoint["model_pos"]
#model_pos_train = load_pretrained_weights(model_pos_train, posetrans_checkpoint)
#################
causal_shift = 0
model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
    model_pos_train = nn.DataParallel(model_pos_train)
    model_pos_train = model_pos_train.cuda()


if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)


test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right,
                                    receptive_field = receptive_field, estimation_field=estimation_field, frames_gap=frames_gap)

print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

# =============================================================================
# def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
#     inputs_2d_p = torch.squeeze(inputs_2d)
#     inputs_3d_p = inputs_3d.permute(1,0,2,3)
#     out_num = inputs_2d_p.shape[0] - receptive_field + 1
#     eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
#     for i in range(out_num):
#         eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
#     return eval_input_2d, inputs_3d_p
# =============================================================================

def eval_data_prepare_3d_3d(frames_gap,receptive_field,estimation_field,input_3d,outputs_3d,divide_data):
    input_3d_p = torch.squeeze(input_3d)
    output_3d_p = torch.squeeze(outputs_3d)  
    divide_data = divide_data*150
    out_num = (input_3d_p.shape[0] - receptive_field * frames_gap -estimation_field*frames_gap + 1)//divide_data
    out_num = max(out_num,1)
    eval_input_3d = torch.empty(out_num, receptive_field , input_3d_p.shape[1], input_3d_p.shape[2])
    eval_output_3d = torch.empty(out_num, estimation_field , output_3d_p.shape[1], output_3d_p.shape[2])
    
    for i in range(out_num):
        current_frames_in = input_3d_p[ i*divide_data: i*divide_data + receptive_field*frames_gap]  
        eval_input_3d[i,:,:,:] = current_frames_in[::frames_gap, :, :]
        current_frames_out = output_3d_p[ (receptive_field-1)*frames_gap + i*divide_data : (receptive_field-1)*frames_gap  + i*divide_data + estimation_field*frames_gap]
        eval_output_3d[i,:,:,:] = current_frames_out[::frames_gap, :, :]
    return eval_input_3d, eval_output_3d


def eval_data_prepare_render(frames_gap,receptive_field,estimation_field,input_3d,outputs_3d,divide_data):
    input_3d_p = torch.squeeze(input_3d)
    output_3d_p = torch.squeeze(outputs_3d)  
#    divide_data = divide_data*150
    out_num = (input_3d_p.shape[0] - receptive_field * frames_gap + 1)//divide_data
    out_num = max(out_num,1)
    eval_input_3d = torch.empty(out_num, receptive_field , input_3d_p.shape[1], input_3d_p.shape[2])
    eval_output_3d = torch.empty(out_num, estimation_field , output_3d_p.shape[1], output_3d_p.shape[2])
    
    for i in range(out_num):
        current_frames_in = input_3d_p[ i*divide_data: i*divide_data + receptive_field*frames_gap]  
        eval_input_3d[i,:,:,:] = current_frames_in[::frames_gap, :, :]
        current_frames_out = output_3d_p[ i*divide_data : i*divide_data + estimation_field*frames_gap]
        eval_output_3d[i,:,:,:] = current_frames_out[::frames_gap, :, :]
    return eval_input_3d, eval_output_3d

def eval_data_prepare_eval(frames_gap,receptive_field,estimation_field,input_3d,outputs_3d,divide_data):
    input_3d_p = torch.squeeze(input_3d)
    output_3d_p = torch.squeeze(outputs_3d)  
    out_num = (input_3d_p.shape[0] - receptive_field * frames_gap -estimation_field*frames_gap + 1)//divide_data
    out_num = max(out_num,1)
    eval_input_3d = torch.empty(out_num, receptive_field , input_3d_p.shape[1], input_3d_p.shape[2])
    eval_output_3d = torch.empty(out_num, estimation_field , output_3d_p.shape[1], output_3d_p.shape[2])
    
    for i in range(out_num):
        current_frames_in = input_3d_p[ i*divide_data: i*divide_data + receptive_field*frames_gap]  
        eval_input_3d[i,:,:,:] = current_frames_in[::frames_gap, :, :]
        current_frames_out = output_3d_p[ (receptive_field-1)*frames_gap + i*divide_data : (receptive_field-1)*frames_gap  + i*divide_data + estimation_field*frames_gap]
        eval_output_3d[i,:,:,:] = current_frames_out[::frames_gap, :, :]
    return eval_input_3d, eval_output_3d


if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    
    lr = args.learning_rate
  #  lr = args.learning_rate/10
    optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=weight_decay)

    lr_decay = args.lr_decay
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    train_generator = ChunkedGenerator(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right,
                                       divide_data=divide_data,receptive_field = receptive_field, estimation_field=estimation_field,frames_gap = frames_gap)
    
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False,
                                              receptive_field = receptive_field, estimation_field=estimation_field, frames_gap=frames_gap)

        
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))

    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

        lr = checkpoint['lr']


    print('** Note: reported losses are averaged over all frames.')
    print('** The final evaluation will be carried out after the last training epoch.')


    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        model_pos_train.train()
        counter = 0
        
        #### train start
        for cameras_train, batch_3d, batch_2d, batch_3d_train, batch_3d_gt in train_generator.next_epoch():
            counter +=1
            cameras_train = torch.from_numpy(cameras_train.astype('float32'))
            new_input_3d = torch.from_numpy(batch_3d_train.astype('float32'))
            new_output_3d = torch.from_numpy(batch_3d_gt.astype('float32'))
     #       new_input_3d += (0.05**0.5)*torch.randn(new_input_3d.shape)
     #       new_output_3d += (0.05**0.5)*torch.randn(new_output_3d.shape)
            
            if torch.cuda.is_available():

                cameras_train = cameras_train.cuda()
                new_input_3d = new_input_3d.cuda()
                new_output_3d = new_output_3d.cuda()
#            inputs_traj = inputs_3d[:, :, :1].clone()
            
            ## To make the first frame az global origin
            new_input_3d[:,:,0] = 0
            new_output_3d[:,:,0] = 0
            
            optimizer.zero_grad()
            new_output_3d = new_output_3d.permute(1,0,2,3)
            new_output_3d_decoder = new_output_3d[:estimation_field].permute(1,0,2,3)
            new_output_3d_gt = new_output_3d[1:].permute(1,0,2,3)
            # Predict 3D poses
            predicted_seq_3d_pos = model_pos_train(new_input_3d,new_output_3d_decoder)
            del new_input_3d
            torch.cuda.empty_cache()

            if new_output_3d_decoder.shape[0] != args.batch_size or new_output_3d_decoder.shape[1] != estimation_field  or new_output_3d_decoder.shape[2]!=17 or new_output_3d_decoder.shape[3]!=3:
                print("wrong shape of the batch!!!!")
            
            loss_3d_pos = seq_mpjpe(predicted_seq_3d_pos, new_output_3d_gt)
            epoch_loss_3d_train += new_output_3d_gt.shape[0] * new_output_3d_gt.shape[1] * loss_3d_pos.item()
            N += new_output_3d_gt.shape[0] * new_output_3d_gt.shape[1]
            if pd.isna(epoch_loss_3d_train):
                print("Nan!!!")
            loss_total = loss_3d_pos
            loss_total.backward()
            optimizer.step()
            del new_output_3d,new_output_3d_decoder, new_output_3d_gt, loss_3d_pos, predicted_seq_3d_pos
            torch.cuda.empty_cache()
            

        losses_3d_train.append(epoch_loss_3d_train / N)
        
        torch.cuda.empty_cache()
        print("training of epoch "+str(epoch) +" is done with "+str(counter)+" loops")
        training_time = (time()-start_time)/60        
#################
        if epoch % 1 == 0:

            # End-of-epoch evaluation
            with torch.no_grad():
                model_pos.load_state_dict(model_pos_train.state_dict(), strict=False)
                model_pos.eval()
    
                epoch_loss_3d_valid = 0
                epoch_loss_traj_valid = 0
                epoch_loss_2d_valid = 0
                N = 0
                if not args.no_eval:
                    # Evaluate on test set
                    for cam, batch, batch_2d, batch_3d_val in test_generator.next_epoch():
                        
                        new_outputs_3d = torch.from_numpy(batch.astype('float32'))
                        new_input_3d = torch.from_numpy(batch_3d_val.astype('float32'))   
                        
                        new_input_3d, new_outputs_3d = eval_data_prepare_3d_3d(frames_gap, receptive_field, estimation_field, new_input_3d, new_outputs_3d, divide_data)

                        if torch.cuda.is_available():
                            new_input_3d = new_input_3d.cuda()
                            new_outputs_3d = new_outputs_3d.cuda()
                        
                        new_outputs_3d[:, :, 0] = 0   
                        new_input_3d[:, :, 0] = 0

                        predicted_3d_pos = model_pos(new_input_3d,new_outputs_3d)

                        del new_input_3d

                        torch.cuda.empty_cache()

                        loss_3d_pos = seq_mpjpe(predicted_3d_pos, new_outputs_3d)
                        epoch_loss_3d_valid += new_outputs_3d.shape[0] * new_outputs_3d.shape[1] * loss_3d_pos.item()
                        N += new_outputs_3d.shape[0] * new_outputs_3d.shape[1]

                        del new_outputs_3d, loss_3d_pos, predicted_3d_pos

                        torch.cuda.empty_cache()

                    losses_3d_valid.append(epoch_loss_3d_valid / N)

                # Evaluate on training set, this time in evaluation mode
                    epoch_loss_3d_train_eval = 0
                    epoch_loss_traj_train_eval = 0
                    epoch_loss_2d_train_labeled_eval = 0
                    N = 0
                
                    for cam, batch, batch_2d ,batch_3d_test in train_generator_eval.next_epoch():
                        if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                            continue

                        new_outputs_3d_test = torch.from_numpy(batch.astype('float32'))                    
                        new_input_3d_test = torch.from_numpy(batch_3d_test.astype('float32'))                    
                    
                        new_input_3d_test, new_outputs_3d_test = eval_data_prepare_3d_3d(frames_gap, receptive_field, estimation_field, new_input_3d_test, new_outputs_3d_test,divide_data)

                        if torch.cuda.is_available():
                            new_input_3d_test = new_input_3d_test.cuda()
                            new_outputs_3d_test = new_outputs_3d_test.cuda()

                        new_input_3d_test[:, :, 0] = 0
                        new_outputs_3d_test[:, :, 0] = 0

                    # Compute 3D poses
                        predicted_3d_pos = model_pos(new_input_3d_test,new_outputs_3d_test)

                        del new_input_3d_test
                        torch.cuda.empty_cache()

                        loss_3d_pos = seq_mpjpe(predicted_3d_pos, new_outputs_3d_test)
                        epoch_loss_3d_train_eval += new_outputs_3d_test.shape[0] * new_outputs_3d_test.shape[1] * loss_3d_pos.item()
                        N += new_outputs_3d_test.shape[0] * new_outputs_3d_test.shape[1]

                        del new_outputs_3d_test, loss_3d_pos, predicted_3d_pos
                        torch.cuda.empty_cache()

                    losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

                # Evaluate 2D loss on unlabeled training set (in evaluation mode)
                epoch_loss_2d_train_unlabeled_eval = 0
                N_semi = 0

#        else:
#            losses_3d_valid.append(epoch_loss_3d_valid / N)
#            losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] total_time %.2f training_time %.2f lr %f 3d_train %f' % (
                epoch + 1,
                elapsed,
                training_time,
                lr,
                losses_3d_train[-1] * 1000))
        else:

            print('[%d] total_time %.2f training_time %.2f lr %f 3d_train %f 3d_eval %f 3d_test %f' % (
                epoch + 1,
                elapsed,
                training_time,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_train_eval[-1] * 1000,
                losses_3d_valid[-1] * 1000))

        # learning rate warmup
#        if epoch < 10:
#            lr = (epoch+1)*args.learning_rate/10
#            for param_group in optimizer.param_groups:
#                param_group['lr'] = lr
        # Decay learning rate exponentially
#        else:
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        # momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        # model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, chk_path)

        #### save best checkpoint
        best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin'.format(epoch))
#        if losses_3d_valid[-1] * 1000 < min_loss:

        if losses_3d_valid[-1] * 1000 < min_loss:
            min_loss = losses_3d_valid[-1] * 1000
            print("save best checkpoint")
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, best_chk_path)
         ###   
        print("save epoch")
        last_chk_path = os.path.join(args.checkpoint, 'last_epoch.bin'.format(epoch))
        torch.save({
            'epoch': epoch,
            'lr': lr,
            'random_state': train_generator.random_state(),
            'optimizer': optimizer.state_dict(),
            'model_pos': model_pos_train.state_dict(),
            # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
            # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
        }, last_chk_path)
        
        ###
        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 1:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(1, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[1:],'--',  color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[1:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[1:],color='C1')
            plt.plot(epoch_x, min(losses_3d_valid)*np.ones(len(losses_3d_train[1:])), color = 'r', linestyle = '--')
            plt.text(2.85,min(losses_3d_valid[1:]), "{:.3f}".format(min(losses_3d_valid)), color="red",ha="right", va="center")
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
          #  plt.xlabel('Epoch')
            plt.xlabel('b '+str(args.batch_size)+' dd '+str(divide_data)+' emb '+str(embed_dim_ratio)+' dep '+str(depth)+' head '+str(num_heads)+' mlp '+str(mlp_ratio)+' drate '+str(drop_rate)+' adrate '+str(attn_drop_rate)+' dprate '+str(drop_path_rate)+' wd '+str(weight_decay))
            plt.xlim((3, epoch))
   #         plt.ylim([0,0.4])
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

            plt.close('all')


# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        # else:
            # model_traj.eval()
        N = 0
        for _, batch, batch_2d, batch_3d_test in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
     #       inputs_3d = torch.from_numpy(batch.astype('float32'))
            new_outputs_3d = torch.from_numpy(batch.astype('float32'))     
            new_input_3d = torch.from_numpy(batch_3d_test.astype('float32'))                    

            new_input_3d_flip = new_input_3d.clone()                   
            new_input_3d_flip[:,:,:,0] *= -1
            new_input_3d_flip[:, :,kps_left + kps_right, :] = new_input_3d_flip[:,:, kps_right + kps_left, :]

            new_outputs_3d_flip = new_outputs_3d.clone()                   
            new_outputs_3d_flip[:,:,:,0] *= -1
            new_outputs_3d_flip[:, :,kps_left + kps_right, :] = new_outputs_3d_flip[:,:, kps_right + kps_left, :]

            new_input_3d, new_outputs_3d = eval_data_prepare_render(frames_gap, receptive_field, estimation_field, new_input_3d, new_outputs_3d,divide_data)
            new_input_3d_flip, new_outputs_3d_flip = eval_data_prepare_render(frames_gap, receptive_field, estimation_field, new_input_3d_flip, new_outputs_3d_flip,divide_data)

            ##### convert size

            if torch.cuda.is_available():
                new_input_3d = new_input_3d.cuda()
                new_input_3d_flip = new_input_3d_flip.cuda()
                new_outputs_3d = new_outputs_3d.cuda()
                new_outputs_3d_flip = new_outputs_3d_flip.cuda()
                
            new_outputs_3d_flip[:, :, 0] = 0
            new_outputs_3d[:, :, 0] = 0   
            new_input_3d_flip[:, :, 0] = 0
            new_input_3d[:, :, 0] = 0

            predicted_3d_pos = model_pos(new_input_3d,new_outputs_3d)
            predicted_3d_pos_flip = model_pos(new_input_3d_flip,new_outputs_3d_flip)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]

     #       predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
      #                                    keepdim=True)

            predicted_3d_pos = (predicted_3d_pos+ predicted_3d_pos_flip)/2

            del  new_input_3d, new_input_3d_flip
            torch.cuda.empty_cache()

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()
            

            error = seq_mpjpe(predicted_3d_pos, new_outputs_3d)
            epoch_loss_3d_pos_scale += new_outputs_3d.shape[0]*new_outputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, new_outputs_3d).item()

            epoch_loss_3d_pos += new_outputs_3d.shape[0]*new_outputs_3d.shape[1] * error.item()
            N += new_outputs_3d.shape[0] * new_outputs_3d.shape[1]

            inputs = new_outputs_3d.cpu().numpy().reshape(-1, new_outputs_3d.shape[-2], new_outputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, new_outputs_3d.shape[-2], new_outputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += new_outputs_3d.shape[0]*new_outputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += new_outputs_3d.shape[0]*new_outputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev


if args.eval_de:
    print('Evaluating...')
    
    def compute_diversity(pred):
        if pred.shape[0] == 1:
            return 0.0
        dist = pdist(pred.cpu().reshape(pred.shape[0], -1))
        diversity = dist.mean().item()
        return diversity
    
    def compute_ade(pred, gt):
        diff = pred - gt
        dist = np.linalg.norm(diff.cpu(), axis=2).mean(axis=1)
        return dist.min()
    
    def compute_fde(pred, gt):
        diff = pred - gt
        dist = np.linalg.norm(diff.cpu(), axis=2)[:, -1]
        return dist.min()
    
    def compute_stats(pred, gt):
        
        return compute_diversity(pred),compute_ade(pred, gt),compute_fde(pred, gt)
        
    def compute_diversity2(pred):
        if pred.shape[0] == 1:
            return 0.0
        dist = pdist(pred.cpu().reshape(pred.shape[0], -1))
        diversity = dist.mean().item()
        return diversity
    
    def compute_ade2(pred, gt):
        diff = pred - gt
        dist = np.linalg.norm(diff.cpu(), axis=2).mean(axis=1).mean(axis=0)
        return dist.min()
    
    def compute_fde2(pred, gt):
        diff = pred - gt
        dist = np.linalg.norm(diff.cpu(), axis=2)[:, -1].mean(axis=0)
        return dist.min()
    
    def compute_stats2(pred, gt):
        
        return compute_diversity2(pred),compute_ade2(pred, gt),compute_fde2(pred, gt)

    def compute_stats3(pred, gt, mrt):
        _,f,_ = gt.shape
        diff = pred - gt
        pos_errors = []
        for i in range(f):
            err = diff[:,i]
            err = np.linalg.norm(err, axis=1).mean()
            pos_errors.append(err)
        return pos_errors
    
    def compute_ade3(pred, gt):
        diff = pred - gt
        dist = np.linalg.norm(diff, axis=2).mean(axis=1).mean(axis=0)
        return dist.min()

    def compute_fde3(pred, gt):
        diff = pred - gt
        dist = np.linalg.norm(diff, axis=2)[:, -1].mean(axis=0)
        return dist.min() 
        
    subjects_test = args.subjects_test.split(',')
    subjects_train = args.subjects_train.split(',')
    #subjects_test = args.subjects_test.split(',')
    #subjects_train = ['S11']
    
    cameras_test, poses_test, poses_test_2d = fetch(subjects_test, action_filter)
    
    test_generator = UnchunkedGenerator(cameras_test, poses_test, poses_test_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right,
                                        receptive_field = receptive_field, estimation_field=estimation_field, frames_gap=frames_gap,render=False)
    
    use_trajectory_model= False
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        # else:
            # model_traj.eval()
        N = 0
        stats = []
        tot_div = 0
        tot_ade = 0
        tot_fde = 0  
        tot_div2 = 0
        tot_ade2 = 0
        tot_fde2 = 0  
        batches = 0
        my_counter=0
        total_errors=0
        total_ade=0
        total_fde=0

        idx = 0

        for _, batch, batch_2d, batch_3d_test in test_generator.next_epoch():
            idx += 1
    
            new_outputs_3d = torch.from_numpy(batch.astype('float32'))     
            new_input_3d = torch.from_numpy(batch_3d_test.astype('float32'))                    
    
            new_input_3d, new_outputs_3d = eval_data_prepare_eval(frames_gap, receptive_field, estimation_field, new_input_3d, new_outputs_3d, divide_data)
    
            ##### convert size
    
            if torch.cuda.is_available():
                new_input_3d = new_input_3d.cuda()
                new_outputs_3d = new_outputs_3d.cuda()
                
            new_outputs_3d[:, :, 0] = 0   
            new_input_3d[:, :, 0] = 0
    
            predicted_3d_pos = model_pos(new_input_3d,new_outputs_3d)
    
            del  new_input_3d
            torch.cuda.empty_cache()
    
            b,e,j,d = predicted_3d_pos.shape
            predicted_3d_pos = predicted_3d_pos.reshape(b,e,-1)
            new_outputs_3d = new_outputs_3d.reshape(b,e,-1)
            c_div2,c_ade2,c_fde2 = compute_stats2(predicted_3d_pos,new_outputs_3d)
            tot_div2 = tot_div2 + c_div2
            tot_ade2 = tot_ade2 + c_ade2
            tot_fde2 = tot_fde2 + c_fde2
            my_counter +=1


            errors = compute_stats3(predicted_3d_pos.cpu(),new_outputs_3d.cpu(),2)
            ADE = compute_ade(predicted_3d_pos.cpu(),new_outputs_3d.cpu())
            FDE = compute_fde(predicted_3d_pos.cpu(),new_outputs_3d.cpu())
            total_errors += np.array(errors)      
            total_ade += ADE
            total_fde += FDE  

        dt = 2/20
    
        print("result of evaluation on data")
        
        for i in range(20):
            print(str((i+1)*dt)[:5]+"  ", end=" ")
        print(" ")
        for i in range(20):
            print(str(total_errors[i]/idx)[:5], end=" ")
        print(" ")
        

        avg_ADE = total_ade 
        avg_FDE = total_fde 
        print(avg_ADE,avg_FDE)



            
            
    # =============================================================================
    #             for i in range(b):
    #                 c_div,c_ade,c_fde = compute_stats(predicted_3d_pos[i].reshape(1,e,-1),new_outputs_3d[i].reshape(1,e,-1))
    #                 tot_div = tot_div + c_div
    #                 tot_ade = tot_ade + c_ade
    #                 tot_fde = tot_fde + c_fde
    # =============================================================================
                
       #         batches = batches + b
    
        avg_div2=tot_div2/my_counter
        avg_ade2=tot_ade2/my_counter
        avg_fde2=tot_fde2/my_counter
        
    # =============================================================================
    #         avg_div = tot_div / batches
    #         avg_ade = tot_ade / batches
    #         avg_fde = tot_fde / batches         
    # =============================================================================
    
    # =============================================================================
    #         print("result of evaluation")
    #         print(avg_div,avg_ade,avg_fde)
    # =============================================================================
    
    
        print("result of evaluation on S11")
        print(avg_div2,avg_ade2,avg_fde2)
        print()
        
if args.render:
    print('Rendering...')

    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    gen = UnchunkedGenerator(None, [ground_truth], [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right,
                             receptive_field = receptive_field, estimation_field=estimation_field, frames_gap=frames_gap,render=True)
    
    prediction = evaluate(gen, return_predictions=True)
    # if model_traj is not None and ground_truth is None:
    #     prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
    #     prediction += prediction_traj

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)

    if args.viz_output is not None:
        if ground_truth is not None:
            # Reapply trajectory

            out_num = (ground_truth.shape[0] - estimation_field * frames_gap + 1)//divide_data
            new_ground_truth = np.zeros((out_num,estimation_field,ground_truth.shape[1],ground_truth.shape[2]))
            for i in range(out_num):
                current_gt = ground_truth[i*divide_data:i*divide_data+estimation_field*frames_gap]
                new_ground_truth[i] = current_gt[::frames_gap]
                
            ground_truth = new_ground_truth
            trajectory = ground_truth[:, :,:1]
            ground_truth[:,:,1:] += trajectory

            [s1,s2,s3,s4] = trajectory.shape

        #    trajectory = np.reshape(trajectory,[s1,1,s2,s3])
            prediction = prediction[:out_num]
            prediction += trajectory
         
            ground_truth = ground_truth.astype('float32')
                  
        #####
        ground_truth = ground_truth[::estimation_field]
        ground_truth = ground_truth.reshape(out_num,num_joints,3)
        
        prediction = prediction[::estimation_field]
        prediction = prediction.reshape(out_num,num_joints,3)
        
        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        if ground_truth is not None:
            prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                    break
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None and not args.viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

        from common.visualization import render_animation

   #     render_animation(input_keypoints, keypoints_metadata, anim_output,
   #                      dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
   #                      limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
   #                      input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
   #                      input_video_skip=args.viz_skip)

        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), 25, args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip)
else:
    print('Evaluating...')
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))

    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)):  # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_poses_3d, out_poses_2d


    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                     pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,joints_right=joints_right,
                                     receptive_field = receptive_field, estimation_field=estimation_field, frames_gap=frames_gap)
            
            e1, e2, e3, ev = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')


    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')
