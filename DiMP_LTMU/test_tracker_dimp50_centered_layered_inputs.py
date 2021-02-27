import os
from run_tracker_centered_layered_inputs import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test DiMP_LTMU
p = p_config()
p.save_results = True

p.use_mask = False
p.name = 'DiMP50_LTMU'

dataset = 'cdtb_colormap'

p.dtype = 'centered_colormap'
p.depth_threshold = None


p.minimun_area_threshold = 0.1    # area changes compared to init_area
p.area_scale_threshold = 1.5      # area changes compared to prev_avg_area
p.conf_threshold = 0.5            # Consider the prediction reliable
p.target_depth_changes_threshold = 1.2 # 1.2
p.conf_rollback = 0.95
p.area_rollback = 0.9
p.rollback_iter = 50

p.radius = 500

p.visualization = False
p.grabcut_visualization = False
p.grabcut_extra = 50
p.grabcut_rz_factor = 1.5       # 1 / 1.5
p.grabcut_rz_threshold = 300
p.grabcut_iter = 2

p.minimun_target_pixels = 16

eval_tracking(dataset, p=p, tracker_name='dimp', tracker_params='dimp50_colormap')
