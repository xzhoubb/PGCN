import torch.utils.data as data

import os
import os.path
import numpy as np
from numpy.random import randint
from ops.I3D_Pooling import I3D_Pooling
from ops.io import load_proposal_file
from ops.utils import temporal_iou
from ops.detection_metrics import segment_tiou
from ops.detection_metrics import segment_distance
from tqdm import tqdm
import time
import pickle
import torch
import math


class PGCNInstance:

    def __init__(self, start_frame, end_frame, video_frame_count,
                 fps=1, label=None,
                 best_iou=None, overlap_self=None):
        self.start_frame = start_frame # 160
        self.end_frame = min(end_frame, video_frame_count) # 346
        self._label = label # 4
        self.fps = fps  # 1

        self.coverage = (end_frame - start_frame) / video_frame_count # 0.11832

        self.best_iou = best_iou # 1.0
        self.overlap_self = overlap_self # 0.0

        self.loc_reg = None
        self.size_reg = None

    def compute_regression_targets(self, gt_list, fg_thresh): # fg_thresh 0.7
        '''
           get regression target of the best iou gt
                loc_reg: center shift propotional to the proposal duration
                size_reg: logarithm of the groundtruth duration over proposal duraiton
        '''
        
        if self.best_iou < fg_thresh:
            # background proposals do not need this
            return
        # find the groundtruth instance with the highest IOU
        ious = [temporal_iou((self.start_frame, self.end_frame), (gt.start_frame, gt.end_frame)) for gt in gt_list]
        best_gt_id = np.argmax(ious)
        best_gt = gt_list[best_gt_id]
        prop_center = (self.start_frame + self.end_frame) / 2
        gt_center = (best_gt.start_frame + best_gt.end_frame) / 2
        prop_size = self.end_frame - self.start_frame + 1
        gt_size = best_gt.end_frame - best_gt.start_frame + 1

        # get regression target:
        # (1). center shift propotional to the proposal duration
        # (2). logarithm of the groundtruth duration over proposal duraiton

        self.loc_reg = (gt_center - prop_center) / prop_size
        try:
            self.size_reg = math.log(gt_size / prop_size)
        except:
            print(gt_size, prop_size, self.start_frame, self.end_frame)
            raise

    @property
    def start_time(self):
        return self.start_frame / self.fps

    @property
    def end_time(self):
        return self.end_frame / self.fps

    @property
    def label(self):
        return self._label if self._label is not None else -1

    @property
    def regression_targets(self):
        return [self.loc_reg, self.size_reg] if self.loc_reg is not None else [0, 0]


class PGCNVideoRecord:
    def __init__(self, prop_record):
        '''
        prop_record: ('vid', num_frames, gt_list, prop_list)
            'vid': '0004bf664eee696a9032c4a2770e12dc'
            num_framse：2995
            gt_list：[['13', '2072', '2140'], ['13', '2224', '2307'], ...]
            prop_list：[['13', '0.78125', '0.8522727272727273', '2232', '2320'], ['13', '0.9411764705882353', '1.0', '2072', '2136'], ...]
        '''
        self._data = prop_record

        frame_count = int(self._data[1]) # 1572

        # build instance record
        self.gt = [
            PGCNInstance(int(x[1]), int(x[2]), frame_count, label=int(x[0]), best_iou=1.0) for x in self._data[2]
            if int(x[2]) > int(x[1])
        ]

        self.gt = list(filter(lambda x: x.start_frame < frame_count, self.gt))

        self.proposals = [
            PGCNInstance(int(x[3]), int(x[4]), frame_count, label=int(x[0]),
                        best_iou=float(x[1]), overlap_self=float(x[2])) for x in self._data[3] if int(x[4]) > int(x[3])
        ]

        self.proposals = list(filter(lambda x: x.start_frame < frame_count, self.proposals))

    @property
    def id(self):
        return self._data[0].strip("\n").split("/")[-1]
    @property
    def num_frames(self):
        return int(self._data[1])

    def get_fg(self, fg_thresh, with_gt=True): # fg_thresh-->0.7, p.best_iou-->the best iou of certain prop between gts
        '''
        return one video foreground props, and cal loc_reg and size_reg of every prop
        '''
        fg = [p for p in self.proposals if p.best_iou > fg_thresh]
        if with_gt:
            fg.extend(self.gt)

        for x in fg:
            x.compute_regression_targets(self.gt, fg_thresh)
        return fg

    def get_negatives(self, incomplete_iou_thresh, bg_iou_thresh,
                      bg_coverage_thresh=0.01, incomplete_overlap_thresh=0.7):
        '''
            incomplete_iou_thresh: 0.3
            bg_iou_thresh: 0.01
            bg_coverage_thresh: 0.02 , background prop coverage (end_f-start_f)/video_num_f)
            incomplete_overlap_thresh: 0.01
        '''

        tag = [0] * len(self.proposals)

        incomplete_props = []
        background_props = []

        for i in range(len(tag)):
            if self.proposals[i].best_iou < incomplete_iou_thresh \
                    and self.proposals[i].overlap_self > incomplete_overlap_thresh:
                tag[i] = 1 # incomplete
                incomplete_props.append(self.proposals[i])

        for i in range(len(tag)):
            if tag[i] == 0 and \
                self.proposals[i].best_iou < bg_iou_thresh and \
                            self.proposals[i].coverage > bg_coverage_thresh:
                background_props.append(self.proposals[i])
        return incomplete_props, background_props


class PGCNDataSet(data.Dataset):

    def __init__(self, dataset_configs, graph_configs, prop_file, prop_dict_path, ft_path, exclude_empty=True,
                 epoch_multiplier=1, test_mode=False, gt_as_fg=True, reg_stats=None):

        self.ft_path = ft_path   # 'data/THUMOS14/I3D_video_level/Rgb_TrainPJ2_All'
        self.prop_file = prop_file # 'data/bsn_train_proposal_list.txt'
        self.prop_dict_path = prop_dict_path # 'data/thumos14_train_prop_dict.pkl'

        self.exclude_empty = exclude_empty # True
        self.epoch_multiplier = epoch_multiplier # 1
        self.gt_as_fg = gt_as_fg # True
        self.test_mode = test_mode

        self.fg_ratio = dataset_configs['fg_ratio'] # 1
        self.incomplete_ratio = dataset_configs['incomplete_ratio'] # 6
        self.bg_ratio = dataset_configs['bg_ratio'] # 1
        self.prop_per_video = dataset_configs['prop_per_video'] # 8


        self.fg_iou_thresh = dataset_configs['fg_iou_thresh'] # 0.7
        self.bg_iou_thresh = dataset_configs['bg_iou_thresh'] # 0.01
        self.iou_threshold = dataset_configs['iou_threshold'] # 0.7
        self.dis_threshold = dataset_configs['dis_threshold'] # 0
        self.bg_coverage_thresh = dataset_configs['bg_coverage_thresh'] # 0.02
        self.incomplete_iou_thresh = dataset_configs['incomplete_iou_thresh'] # 0.3
        self.incomplete_overlap_thresh = dataset_configs['incomplete_overlap_thresh'] # 0.01

        self.starting_ratio = dataset_configs['starting_ratio'] # 0.5
        self.ending_ratio = dataset_configs['ending_ratio'] # 0.5


        self.adj_num = graph_configs['adj_num'] # 21
        self.child_num = graph_configs['child_num'] # 4
        self.child_iou_num = graph_configs['iou_num'] # 8
        self.child_dis_num = graph_configs['dis_num'] # 2

        denum = self.fg_ratio + self.bg_ratio + self.incomplete_ratio # 8
        self.fg_per_video = int(self.prop_per_video * (self.fg_ratio / denum)) # 1
        self.bg_per_video = int(self.prop_per_video * (self.bg_ratio / denum)) # 1
        self.incomplete_per_video = self.prop_per_video - self.fg_per_video - self.bg_per_video # 6

        parse_time = time.time()
        self._parse_prop_file(stats=reg_stats)
        print("File parsed. Time:{:.2f}".format(time.time() - parse_time))

        """pre-compute iou and distance among proposals"""
        if os.path.exists(self.prop_dict_path):
            construct_time = time.time()
            # if "val" not in self.prop_dict_path:
            dicts = pickle.load(open(self.prop_dict_path, "rb"))
            print("Dict constructed. Time:{:.2f}".format(time.time() - construct_time))
            self.act_iou_dict, self.act_dis_dict, self.prop_dict = dicts[0], dicts[1], dicts[2]
        else:
            self.prop_dict = {}
            self.act_iou_dict = {}
            self.act_dis_dict = {}
            construct_time = time.time()
            if self.test_mode:
                self._prepare_test_iou_dict() # self.prop_dict: {}; self.act_iou_dict, IOU of video_prop; self.act_dis_dict: distance of video_prop
            else:
                self._prepare_iou_dict()
            print("Dict constructed. Time:{:.2f}".format(time.time() - construct_time))
            
            '''
                self.prop_dict: {'video_validation_0000154': [fg, incomp, bg], ...}
                    fg: (116), incomp(130), bg(323)
                    video_prop = fg + incomp + bg (569)

                self.act_iou_dict: {'video_validation_0000154': iou_array_0, ...}
                    iou_array: (569,569), IOU ratio of video_prop

                self.act_dis_dict: {'video_validation_0000154': distance_array, ...}
                    distance_array: (569,569), distance of video_prop
            '''
            pickle.dump([self.act_iou_dict, self.act_dis_dict, self.prop_dict], open(self.prop_dict_path, "wb"))

    def _prepare_iou_dict(self):
        '''
            generate self.prop_dict, self.act_iou_dict and self.act_dis_dict
        '''
        pbar = tqdm(total=len(self.video_list))
        for cnt, video in enumerate(self.video_list):
            pbar.update(1)
            fg = video.get_fg(self.fg_iou_thresh, self.gt_as_fg)
            incomp, bg = video.get_negatives(self.incomplete_iou_thresh, self.bg_iou_thresh,
                                             self.bg_coverage_thresh, self.incomplete_overlap_thresh)
            self.prop_dict[video.id] = [fg, incomp, bg]
            video_pool = fg + incomp + bg
            
            # calculate act iou matrix
            prop_array = np.array([[prop.start_frame, prop.end_frame] for prop in video_pool])
            iou_array, overlap_array = segment_tiou(prop_array, prop_array)
            # iou_array: 2-dim array [m x n] with IOU ratio.
            # overlap_array: 2-dim array [m x n] with overlap(interation of frame eg.166)
            self.act_iou_dict[video.id] = iou_array
            
            # calculate act distance matrix
            prop_array = np.array([[prop.start_frame, prop.end_frame] for prop in video_pool])
            distance_array = segment_distance(prop_array, prop_array) # 2-dim array [m x n] with distance.
            self.act_dis_dict[video.id] = distance_array
        pbar.close()

    def _prepare_test_iou_dict(self):
        pbar = tqdm(total=len(self.video_list))
        for cnt, video in enumerate(self.video_list):
            pbar.update(1)
            video_pool = video.proposals
            # calculate act iou matrix
            prop_array = np.array([[prop.start_frame, prop.end_frame] for prop in video_pool])
            iou_array, overlap_array = segment_tiou(prop_array, prop_array)
            self.act_iou_dict[video.id] = iou_array
            # calculate act distance matrix
            prop_array = np.array([[prop.start_frame, prop.end_frame] for prop in video_pool])
            distance_array = segment_distance(prop_array, prop_array)
            self.act_dis_dict[video.id] = distance_array
        pbar.close()

    def _parse_prop_file(self, stats=None):
        '''
            load proposal file and generate self.video_list
        '''
        prop_info = load_proposal_file(self.prop_file)
        '''
            [('vid', num_frames, gt_list, prop_list),...]
            'vid': '0004bf664eee696a9032c4a2770e12dc'
            num_framse：2995
            gt_list：[['13', '2072', '2140'], ['13', '2224', '2307'], ...]
            prop_list：[['13', '0.78125', '0.8522727272727273', '2232', '2320'], ['13', '0.9411764705882353', '1.0', '2072', '2136'], ...]
        '''

        self.video_list = [PGCNVideoRecord(p) for p in prop_info] # len --> 199

        if self.exclude_empty: 
            self.video_list = list(filter(lambda x: len(x.gt) > 0, self.video_list))

        # {'video_validation_0000051': <PGCNVideoRecord.0>, ...}
        self.video_dict = {v.id: v for v in self.video_list} 

        # construct three pools:
        # 1. Foreground
        # 2. Background
        # 3. Incomplete

        self.fg_pool = []
        self.bg_pool = []
        self.incomp_pool = []

        for v in self.video_list:
            self.fg_pool.extend([(v.id, prop) for prop in v.get_fg(self.fg_iou_thresh, self.gt_as_fg)])

            incomp, bg = v.get_negatives(self.incomplete_iou_thresh, self.bg_iou_thresh,
                                         self.bg_coverage_thresh, self.incomplete_overlap_thresh)

            self.incomp_pool.extend([(v.id, prop) for prop in incomp])
            self.bg_pool.extend([(v.id, prop) for prop in bg])

        if stats is None:
            self._compute_regresssion_stats()
            # cal [[loc_reg_mean, size_reg_mean], [loc_reg_std, size_reg_std]] above all fg props
            # array([[-0.0103171 , -0.01709578],[ 0.09711515,  0.19089624]])
        else:
            self.stats = stats




    def _sample_child_nodes(self, video_pool, center_idx, video_id):
        '''
            sample 4 child node from center_idx node(prop)
            1.find top 8 iou largest props from contextual props(iou > 0.7) of center prop
            2.find top 2 small dis/ closet props from surround props(iou<0 and dis > 0) of center prop
            3.random sample 8 node from the 10 node above
        '''
        # obtain iou array for all the proposals
        act_iou_array = self.act_iou_dict[video_id][center_idx, :] # 569
        act_iou_array = np.squeeze(act_iou_array)
        # remove self
        rm_act_iou_array = act_iou_array.copy()
        rm_act_iou_array[center_idx] = 0
        # filter the proposals
        pos_iou_idx = np.where(rm_act_iou_array > self.iou_threshold)[0] # 0.7, (37,)
        if pos_iou_idx.size != 0:
            pos_iou_arr = rm_act_iou_array[pos_iou_idx]
            sorted_pos_iou_idx = np.argsort(-pos_iou_arr).tolist() # large to small id of pos_iou_arr
            selected_pos_iou_idx = np.tile(sorted_pos_iou_idx, self.child_iou_num) # self.child_iou_num 8
            ref_iou_idx = selected_pos_iou_idx[:self.child_iou_num] # get top 8 large iou id
            abs_iou_idx = pos_iou_idx[ref_iou_idx]
        else:
            abs_iou_idx = np.tile(np.array(center_idx), self.child_iou_num) # if no enough node , self

        # obtain dis array for all the proposals
        act_dis_array = self.act_dis_dict[video_id][center_idx, :]
        act_dis_array = np.squeeze(act_dis_array)
        selected_ious_ind = act_iou_array <= 0
        selected_dis_ind = act_dis_array > self.dis_threshold # dis > 0
        selected_ind = np.logical_and(selected_ious_ind, selected_dis_ind) # iou <= 0
        pos_dis_idx = np.where(selected_ind == 1)[0]
        if pos_dis_idx.size != 0:
            pos_dis_arr = act_dis_array[pos_dis_idx]
            sorted_pos_dis_idx = np.argsort(pos_dis_arr).tolist() # small to large dis id
            selected_pos_dis_idx = np.tile(sorted_pos_dis_idx, self.child_dis_num) # self.child_dis_num 2
            ref_dis_idx = selected_pos_dis_idx[:self.child_dis_num] # get top 2 small dis / top 2 close id
            abs_dis_idx = pos_dis_idx[ref_dis_idx]
        else:
            abs_dis_idx = np.tile(np.array(center_idx), self.child_dis_num)

        # obtain child idxs
        abs_child_idx = np.concatenate([abs_iou_idx, abs_dis_idx])
        np.random.shuffle(abs_child_idx)
        abs_child_idx = abs_child_idx[:self.child_num] # 4 child

        return [video_pool[ind] for ind in abs_child_idx], \
               [ind for ind in abs_child_idx]

    def _sample_proposals_via_graph(self, center_prop, video_id, proposal_type, video_pool, abs_center_idx):
        '''
           graph: 1 + 4 + 4*4 = 21 node
           center -- > 4 son node
           son node -- > 4 ground son
        '''
        # video_pool is video full pool --- fg + incomp + bg
        prop_idx_list = [abs_center_idx]
        selected_props = [((video_id, center_prop), proposal_type)]

        center_idx = abs_center_idx
        for stage_cnt in range(self.child_num+1): # 5 node
            # sample proposal with the largest iou
            props, idxs = self._sample_child_nodes(video_pool, center_idx, video_id) # get 4 child node
            prop_idx_list.extend(idxs)
            for prop in props:
                selected_props.append(((video_id, prop), proposal_type))

            center_idx = prop_idx_list[stage_cnt+1] # use the 4 son node of the fisrt center as center

        return selected_props

    def _sample_adjacent_proposals(self, proposal_type, video_id, type_pool, requested_num, video_full_pool,
                                   video_pool_list):
        ref_center_idx = np.random.choice(len(type_pool), requested_num)
        # below code is processed for cal id of video_full_pool
        if proposal_type == 0: # fg
            abs_center_idx = ref_center_idx[0]
        elif proposal_type == 1: # incom
            abs_center_idx = ref_center_idx[0] + len(video_pool_list[0])
        else:# bg
            abs_center_idx = ref_center_idx[0] + len(video_pool_list[0]) + len(video_pool_list[1])
        center_prop = type_pool[ref_center_idx[0]]

        props = self._sample_proposals_via_graph(center_prop, video_id, proposal_type,
                                                 video_full_pool, abs_center_idx)
        return props

    def _video_centric_sampling(self, video):
        '''
            sample 8 prop(1fg+6incomp+1bg) * adjacent_proposals(1center+4son+4*4grandson) = 168 nodes
        '''

        fg, incomp, bg = self.prop_dict[video.id][0], self.prop_dict[video.id][1], self.prop_dict[video.id][2]

        video_full_pool = fg + incomp + bg

        out_props = []
        video_pool_list = [fg, incomp, bg]

        for i in range(self.fg_per_video): # 1
            props = self._sample_adjacent_proposals(0, video.id, fg, 1, video_full_pool, video_pool_list)
            out_props.extend(props)  # sample foreground

        for i in range(self.incomplete_per_video): # 6
            if len(incomp) == 0:
                props = self._sample_adjacent_proposals(0, video.id, fg, 1, video_full_pool, video_pool_list)
            else:
                props = self._sample_adjacent_proposals(1, video.id, incomp, 1, video_full_pool, video_pool_list)
            out_props.extend(props)  # sample incomp

        for i in range(self.bg_per_video): # 1
            if len(bg) == 0:
                props = self._sample_adjacent_proposals(0, video.id, fg, 1, video_full_pool, video_pool_list)
            else:
                props = self._sample_adjacent_proposals(2, video.id, bg, 1, video_full_pool, video_pool_list)
            out_props.extend(props)  # sample bg


        return out_props


    def _sample_indices(self, valid_length, num_seg):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (valid_length + 1) // num_seg
        if average_duration > 0:
            # normal cases
            offsets = np.multiply(list(range(num_seg)), average_duration) \
                             + randint(average_duration, size=num_seg)
        elif valid_length > num_seg:
            offsets = np.sort(randint(valid_length, size=num_seg))
        else:
            offsets = np.zeros((num_seg, ))

        return offsets


    def _sample_pgcn_indices(self, prop, frame_cnt):
        start_frame = prop.start_frame + 1 # 909
        end_frame = prop.end_frame # 1198

        duration = end_frame - start_frame + 1 # 290
        assert duration != 0, (prop.start_frame, prop.end_frame, prop.best_iou)

        valid_starting = max(1, start_frame - int(duration * self.starting_ratio)) # 764, extented start frame
        valid_ending = min(frame_cnt, end_frame + int(duration * self.ending_ratio)) # 1343, extented end frame

        # get starting
        act_s_e = (start_frame, end_frame)
        comp_s_e = (valid_starting, valid_ending)

        offsets = np.concatenate((act_s_e, comp_s_e)) # array([ 909, 1198,  764, 1343])

        return offsets

    def _load_prop_data(self, prop):
        '''
        Input:
            prop: (('video_validation_0000154', <pgcn_dataset.PGCNIn...b93fe2b90>), 0)
        
        Output:
            prop_indices: array([ start_frame, end_frame, entend_start_frame, extend_end_frame])
            label: prop label
            reg_targets: normalized [self.loc_reg, self.size_reg]
            prop[1] : 0 ,prop_type(0-fg, 1-incomp, 2-bg)
        '''
        # read frame count
        frame_cnt = self.video_dict[prop[0][0]].num_frames # 1572

        # sample segment indices
        prop_indices = self._sample_pgcn_indices(prop[0][1], frame_cnt) 

        # get label
        if prop[1] == 0:
            label = prop[0][1].label # fg
        elif prop[1] == 1:
            label = prop[0][1].label  # incomplete
        elif prop[1] == 2:
            label = 0  # background
        else:
            raise ValueError()

        # get regression target
        if prop[1] == 0:
            reg_targets = prop[0][1].regression_targets # [self.loc_reg, self.size_reg]
            # normalize reg_targets
            reg_targets = (reg_targets[0] - self.stats[0][0]) / self.stats[1][0], \
                          (reg_targets[1] - self.stats[0][1]) / self.stats[1][1]
        else:
            reg_targets = (0.0, 0.0)

        return prop_indices, label, reg_targets, prop[1]

    def _compute_regresssion_stats(self):

        targets = []
        for video in self.video_list:
            fg = video.get_fg(self.fg_iou_thresh, False)
            for p in fg:
                targets.append(list(p.regression_targets))

        self.stats = np.array((np.mean(targets, axis=0), np.std(targets, axis=0)))

    def get_test_data(self, video):

        props = video.proposals
        video_id = video.id
        frame_cnt = video.num_frames

        # process proposals to subsampled sequences
        rel_prop_list = []
        proposal_tick_list = []

        for proposal in props:

            rel_prop = proposal.start_frame / frame_cnt, proposal.end_frame / frame_cnt # unit to 1
            rel_duration = rel_prop[1] - rel_prop[0]
            rel_starting_duration = rel_duration * self.starting_ratio
            rel_ending_duration = rel_duration * self.ending_ratio
            rel_starting = rel_prop[0] - rel_starting_duration
            rel_ending = rel_prop[1] + rel_ending_duration

            real_rel_starting = max(0.0, rel_starting)
            real_rel_ending = min(1.0, rel_ending)


            proposal_ticks =  int(rel_prop[0] * frame_cnt), int(rel_prop[1] * frame_cnt), \
                              int(real_rel_starting * frame_cnt), int(real_rel_ending * frame_cnt)

            rel_prop_list.append(rel_prop) 
            # torch.Size([num_prop, 2])
            # [0.7452, 0.7746], [prop_start_real, prop_end_real]
            # prop with unit 1

            proposal_tick_list.append(proposal_ticks) 
            # torch.Size([num_prop, 4])
            # [2232, 2320, 2187, 2364], [prop_start_fr, prop_end_fr, extend_prop_start_fr, extend_prop_end_fr]
            # prop with unit frame

        return torch.from_numpy(np.array(rel_prop_list)), \
               torch.from_numpy(np.array(proposal_tick_list)), \
               video_id, video.num_frames

    def get_training_data(self, index):
        '''
        output: 168 node(prop)
            act_prop_ft：torch.Size([168, 1024]),
            comp_prop_ft： torch.Size([168, 3072])
            out_prop_type： (168,)
            out_prop_labels: (168,)
            out_prop_reg_targets: torch.Size([168, 2])
        '''

        video = self.video_list[index] # <PGCNVideoRecord.0>
        props = self._video_centric_sampling(video) # 168 nodes

        out_prop_ind = []
        out_prop_type = []
        out_prop_labels = []
        out_prop_reg_targets = []

        for idx, p in enumerate(props):
            prop_indices, prop_label, reg_targets, prop_type = self._load_prop_data(p)

            out_prop_ind.append(prop_indices)
            out_prop_labels.append(prop_label)
            out_prop_reg_targets.append(reg_targets)
            out_prop_type.append(prop_type)

        out_prop_labels = torch.from_numpy(np.array(out_prop_labels)) # torch.Size([168])
        out_prop_reg_targets = torch.from_numpy(np.array(out_prop_reg_targets, dtype=np.float32)) # torch.Size([168, 2])
        out_prop_type = torch.from_numpy(np.array(out_prop_type)) # torch.Size([168])

        #load prop fts
        vid_full_name = video.id
        vid = vid_full_name.split('/')[-1] 
        # out_prop_ind: [[s_f, e_f, entend_s_f, extend_e_f],...]
        act_prop_ft, comp_prop_ft = I3D_Pooling(out_prop_ind, vid, self.ft_path, video.num_frames)

        return (act_prop_ft, comp_prop_ft), out_prop_type, out_prop_labels, out_prop_reg_targets

    def get_all_gt(self):
        gt_list = []
        for video in self.video_list:
            vid = video.id
            gt_list.extend([[vid, x.label - 1, x.start_frame / video.num_frames,
                             x.end_frame / video.num_frames] for x in video.gt])
        return gt_list

    def __getitem__(self, index):
        real_index = index % len(self.video_list)
        if self.test_mode:
            return self.get_test_data(self.video_list[real_index])
        else:
            return self.get_training_data(real_index)
            # (act_prop_ft, comp_prop_ft), out_prop_type, out_prop_labels, out_prop_reg_targets

    def __len__(self):
        return len(self.video_list) * self.epoch_multiplier
