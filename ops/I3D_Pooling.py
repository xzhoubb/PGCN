import torch
import os
import numpy as np
from numpy.random import randint
import pandas as pd
import time

def I3D_Pooling(prop_indices, vid, ft_path, n_frame, n_seg=1):
    # n_frame 1572
    #ft_tensor = torch.load(os.path.join(ft_path, vid)).float() # torch.Size([189, 1024])
   
    ft_array = np.load(os.path.join(ft_path, vid+'.npy')) # i3d
    # ft_array = np.load(os.path.join(ft_path, vid+'_32.npy')) # slowfast
    ft_tensor = torch.from_numpy(ft_array).float()

    audio_ft_path = "/home/xzhou3/research/action_detect/datasets/datas/train/vggish_feature"
    #audio_ft_array = np.load(os.path.join(ft_path, vid+'.npy').replace('i3d_feature', 'vggish_feature'))
    audio_ft_array = np.load(os.path.join(audio_ft_path, vid+'.npy'))
    audio_ft_tensor = torch.from_numpy(audio_ft_array).float()
    
    fts_all_act = []
    fts_all_comp = []

    for prop in prop_indices: # (168,)

        act_s = prop[0]
        act_e = prop[1]
        comp_s = prop[2]
        comp_e = prop[3]

        start_ft = feature_pooling(comp_s, act_s, vid,
                                  n_frame, n_seg, 'max', ft_tensor, audio_ft_tensor)
        end_ft = feature_pooling(act_e, comp_e, vid,
                                  n_frame, n_seg, 'max', ft_tensor, audio_ft_tensor)
        act_ft = feature_pooling(act_s, act_e, vid,
                                  n_frame, n_seg, 'max', ft_tensor, audio_ft_tensor)
        comp_ft = [start_ft, act_ft, end_ft]
        comp_ft = torch.cat(comp_ft, dim=0)

        fts_all_act.append(act_ft) # 1024
        fts_all_comp.append(comp_ft) # 3072

    #fts_all_act = torch.stack(fts_all_comp) # torch.Size([168, 1024])
    fts_all_act = torch.stack(fts_all_act) # torch.Size([168, 1024])
    fts_all_comp = torch.stack(fts_all_comp) # torch.Size([168, 3072])

    return fts_all_act, fts_all_comp

def feature_pooling(start_ind, end_ind, vid, n_frame, n_seg, type, ft_tensor, audio_ft_tensor):
    #for turn
    interval = 8
    clip_length = 64

    fts = []
    fts_all = []

    offsets, average_duration = sample_indices(start_ind, end_ind, n_seg) # n_seg means seg of one prop
    # offset:0, average_duration: end-start+1
    ft_num = ft_tensor.size()[0] # 189
    audio_ft_num = audio_ft_tensor.size()[0]
    for off in offsets:

        fts = []

        start_unit = int(min(ft_num-1, np.floor(float(start_ind+off)/interval))) 
        end_unit = int(min(ft_num-2, np.ceil(float(end_ind-clip_length)/interval)))
                   # do not use the last few frames not divisible by 64
        audio_start_unit = int(min((np.floor(audio_ft_num*start_ind/n_frame)), audio_ft_num-1))
        audio_end_unit = int(min((np.ceil(audio_ft_num*end_ind/n_frame)), audio_ft_num-2, ))

        if start_unit < end_unit:
            video_feature_max = torch.max(ft_tensor[start_unit: end_unit+1, :], 0)[0] # (1024,) max_pooling
            video_feature_avg = torch.mean(ft_tensor[start_unit: end_unit+1, :], 0) # (1024,) avg_pooling
            video_feature = video_feature_max + video_feature_avg
            # fts.append(video_feature_max)
            # fts.append(video_feature_avg)
            # fts.append(video_feature_max+video_feature_avg) # max+avg

            # audio_feature
            audio_feature_max = torch.max(audio_ft_tensor[audio_start_unit: audio_end_unit+1, :], 0)[0] # (128,) max
            audio_feature_avg = torch.mean(audio_ft_tensor[audio_start_unit: audio_end_unit+1, :], 0) # (128,) avg
            audio_feature = audio_feature_max + audio_feature_avg

            #fts.append(torch.cat([video_feature, audio_feature], dim=0)) # (1152,)
            fts.append(video_feature_avg) # slowfast(20)
        else:
            #fts.append(torch.cat([ft_tensor[start_unit], audio_ft_tensor[audio_start_unit]], dim=0)) # (1024,)
            fts.append(ft_tensor[start_unit])
        fts_all.append(fts[0])

    fts_all = torch.stack(fts_all)

    return fts_all.squeeze() # torch.Size([1024])

def sample_indices(start, end, num_seg):
    """
    :param record: VideoRecord
    :return: list
    """
    valid_length = end - start # 127 frame
    average_duration = (valid_length + 1) // num_seg
    if average_duration > 0:
        # normal cases
        offsets = np.multiply(list(range(num_seg)), average_duration) # array([0])
    elif valid_length > num_seg:
        offsets = np.sort(randint(valid_length, size=num_seg))
    else:
        offsets = np.zeros((num_seg, ))

    return offsets, average_duration
