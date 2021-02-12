import os
from run_tracker_layered_inputs import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test DiMP_LTMU
p = p_config()
p.save_results = True
p.visualization = False
p.use_mask = False
p.name = 'DiMP50_LTMU'

p.area_max_min_factor = 1.8
p.init_area_scale_factor = 0.1
p.conf_threshold = 0.45
p.dtype = 'layered_colormap' # Dataset[5:] if Dataset in ['cdtb_depth', 'cdtb_colormap','cdtb_rgbd'] else 'rgb'
p.depth_threshold = None
p.conf_rollback = 0.8
p.rollback_iter = 50
p.reset_area_factor = 0.3


'''
    - layeredColormap / layeredDepth : to alleivate the effects of depth changing
        - using the fixed normalization, when depth changes too much, the input depth features change significantly !!!
        - using the layered inputs, H *W * (N*C) :
            - we normalize the depth / colormap with the Low/High boundary of the layer, e.g. 0 - 1000, 1000 - 2000,
              then the normalized depth changes less

            - we search the target in each layer of N layers,
                - the middle i-th layer is from previous prediction, if so, return target position
                - if not , search the (i-1)-th and (i+1)-th layers, until find the target, update the LayerIndex

        - How to automatically calculate the K and layer_bin ??, according to the inital image ??

        - How to train it? use the groundtruth box

        - one problem, when target move form one layer to another layer,
          the normalized depth values changes significantly (two cases):
            - from Minimum 0.05 to Maximum 0.99
            - from Maximum 0.99 to Minimum 0.05

    - is it possible to get the slice of the depth by using the target depth as the center ? -1.0m -> depth <- +1.0m ??

    - should consider the 3D position of the target, (x,y) in RGB, z in depth
'''

dataset = 'cdtb_depth'
p.dtype = 'layered_raw_depth' # layered_colormap, layered_raw_depth, layered_normalized_depth, rgb, rgbd, colormap, raw_depth, layered_raw_depth

eval_tracking(dataset, p=p, tracker_name='dimp', tracker_params='dimp50_depth')
