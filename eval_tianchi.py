import argparse
import numpy as np
from ruamel import yaml

from pgcn_dataset import PGCNDataSet
import pandas as pd
from multiprocessing import Pool
from terminaltables import *

import sys
sys.path.append('./anet_toolkit/Evaluation')
from anet_toolkit.Evaluation.eval_detection import compute_average_precision_detection
import os
import pickle
import json
import time

parser = argparse.ArgumentParser(
    description="PGCN eval Tool")
parser.add_argument('dataset_cfg', type=str)
parser.add_argument('--dataset', type=str, default='tianchi')
parser.add_argument('-j', '--ap_workers', type=int, default=32)
args = parser.parse_args()

configs = yaml.load(open(args.dataset_cfg, 'r'), Loader=yaml.RoundTripLoader)['tianchi']
dataset_configs = configs['dataset_configs']
graph_configs = configs["graph_configs"]
model_configs = configs["model_configs"]
tianchi_result_path = dataset_configs["tianchi_result_path"]
tianchi_map_path = dataset_configs['tianchi_map_path']
tianchi_val_num_frame = dataset_configs['val_num_frame']
num_class = model_configs['num_class']

with open(tianchi_map_path,'r') as f:
    cls_map_dict = json.load(f)

with open(tianchi_val_num_frame,'r') as f:
    vid_num_frame_dict = json.load(f)

with open(tianchi_result_path,'r') as f:
    result_dict = json.load(f)
print ('eval result from: ', tianchi_result_path)

def id_mapping(lable_name):
    for id, name in cls_map_dict.items():
        if name == lable_name:
            return id
    return -1 

def ravel_detections(result_dict, num_class):
    result_list = [[] for i in range(num_class)]

    for vid, dets in result_dict.items():
        for det in dets:
            cls = int(id_mapping(det['label'])) - 1
            temp_num_frame = vid_num_frame_dict[vid]
            t_start = det['segment'][0]*15/temp_num_frame
            t_end = det['segment'][1]*15/temp_num_frame
            score = det['score']
            result_list[cls].append([vid, cls, t_start, t_end, score])

    df_list = []
    for i in result_list:
        df = pd.DataFrame(i, columns=["video-id", "cls","t-start", "t-end", "score"])
        df_list.append(df)
    return df_list

plain_detections = ravel_detections(result_dict, num_class)


dataset = PGCNDataSet(dataset_configs, graph_configs,
                    prop_file=dataset_configs['eval_prop_file'],
                    prop_dict_path=dataset_configs['eval_dict_path'],
                    ft_path=dataset_configs['eval_ft_path'],
                    test_mode=True)

# get gt
all_gt = pd.DataFrame(dataset.get_all_gt(), columns=["video-id", "cls","t-start", "t-end"])
gt_by_cls = []
for cls in range(num_class):
    gt_by_cls.append(all_gt[all_gt.cls == cls].reset_index(drop=True).drop('cls', 1))

# pickle.dump(gt_by_cls, open('gt_dump.pc', 'wb'), pickle.HIGHEST_PROTOCOL)
# pickle.dump(plain_detections, open('pred_dump.pc', 'wb'), pickle.HIGHEST_PROTOCOL)
print("Calling mean AP calculator from toolkit with {} workers...".format(args.ap_workers))

start_time = time.time()

if args.dataset == 'activitynet1.2':
    iou_range = np.arange(0.5, 1.0, 0.05)
elif args.dataset == 'thumos14':
    iou_range = np.arange(0.1, 1.0, 0.1)
elif args.dataset == 'tianchi':
    iou_range = np.arange(0.1, 1.0, 0.1)
else:
    raise ValueError("unknown dataset {}".format(args.dataset))

ap_values = np.empty((num_class, len(iou_range)))


def eval_ap(iou, iou_idx, cls, gt, predition):
    ap = compute_average_precision_detection(gt, predition, iou)
    sys.stdout.flush()
    return cls, iou_idx, ap


def callback(rst):
    sys.stdout.flush()
    ap_values[rst[0], rst[1]] = rst[2][0]

pool = Pool(args.ap_workers)
jobs = []
for iou_idx, min_overlap in enumerate(iou_range):
    for cls in range(num_class):
        jobs.append(pool.apply_async(eval_ap, args=([min_overlap], iou_idx, cls, gt_by_cls[cls], plain_detections[cls],),callback=callback))
pool.close()
pool.join()
print("Evaluation done.\n\n")
map_iou = ap_values.mean(axis=0)
display_title = "Detection Performance on {}".format(args.dataset)

display_data = [["IoU thresh"], ["mean AP"]]

for i in range(len(iou_range)):
    display_data[0].append("{:.02f}".format(iou_range[i]))
    display_data[1].append("{:.04f}".format(map_iou[i]))

display_data[0].append('Average')
display_data[1].append("{:.04f}".format(map_iou.mean()))
table = AsciiTable(display_data, display_title)
table.justify_columns[-1] = 'right'
table.inner_footing_row_border = True
print(table.table)

print("\n\neval time：",time.time()-start_time)