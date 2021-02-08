import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test DiMP_LTMU
p = p_config()
p.save_results = True
p.visualization = True
p.name = 'DiMP18_LTMU'
eval_tracking('cdtb_color', p=p, tracker_name='dimp', tracker_params='dimp18')
