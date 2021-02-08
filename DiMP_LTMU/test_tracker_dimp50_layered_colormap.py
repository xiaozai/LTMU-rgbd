import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test DiMP_LTMU
p = p_config()
p.save_results = True
p.visualization = True
p.name = 'DiMP50_LTMU'

'''
    - layeredColormap / layeredDepth : to alleivate the effects of depth changing
        - using the fixed normalization, when depth changes too much, the input depth features change significantly !!!
        - using the layered inputs, H *W * (N*C) :
            - we normalize the depth / colormap with the Low/High boundary of the layer, e.g. 0 - 1000, 1000 - 2000,
              then the normalized depth changes less
              
            - we search the target in each layer of N layers,
                - the middle i-th layer is from previous prediction, if so, return target position
                - if not , search the (i-1)-th and (i+1)-th layers, until find the target, update the LayerIndex
'''
eval_tracking('cdtb_colormap', p=p, tracker_name='dimp', tracker_params='dimp50_colormap', dtype='layered_colormap')
