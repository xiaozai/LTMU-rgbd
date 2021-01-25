import os
from run_tracker import eval_tracking, p_config
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse

parser = argparse.ArgumentParser(description='Parameters for evaluation')
parser.add_argument('--tracker_name', type=str, default='dimp',
                    help='tracker name, e.g. dimp ')
parser.add_argument('--tracker_params', type=str, default='dimp50_colormap',
                    help='ltr.parameter.trackerName.trackerParams, e.g. dimp50, dimp50_depth, dimp50_colormap')
parser.add_argument('--dataset_name', type=str, default='cdtb_colormap',
                    help='e.g. votlt19, lasot_depth, ...')
parser.add_argument('--video', type=str, default=None,
                    help='the name of video, if only run one sequence')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='0 or 1, or others')
parser.add_argument('--visualization', type=bool, default=True,
                    help='True or False')
parser.add_argument('--use_mask', type=bool, default=False,
                    help='True or False')
parser.add_argument('--save_results', type=bool, default=False,
                    help='True or False')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# test DiMP_LTMU
p = p_config()
p.name = 'DiMP_LTMU_demo'
p.save_results = args.save_results
p.use_mask = args.use_mask
p.visualization = args.visualization
eval_tracking(args.dataset_name, video=args.video, p=p, tracker_name=args.tracker_name, tracker_params=args.tracker_params)
