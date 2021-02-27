import os
import numpy as np
import cv2
import tensorflow as tf
import time
import sys
main_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if main_path not in sys.path:
    sys.path.append(main_path)
    sys.path.append(os.path.join(main_path, 'utils'))
from DiMP_LTMU.Dimp_LTMU import Dimp_LTMU_Tracker
from local_path import lasot_dir, tlp_dir, otb_dir, votlt19_dir, votlt18_dir, cdtb_dir
from tracking_utils import Region
from sklearn.cluster import KMeans
import copy
import matplotlib.pyplot as plt
import scipy
from scipy import stats

class p_config(object):
    Verification = "rtmdnet"
    name = 'Dimp_MU'
    model_dir = 'dimp_mu_votlt'
    checkpoint = 220000
    start_frame = 200
    R_candidates = 20
    save_results = True
    use_mask = True
    save_training_data = False
    visualization = True

    dtype = 'layered_colormap'
    depth_threshold = None

    minimun_area_threshold = 0.1    # area changes compared to init_area
    area_scale_threshold = 1.5     # area changes compared to prev_avg_area
    conf_threshold = 0.5      # Consider the prediction reliable
    target_depth_changes_threshold = 2.0
    conf_rollback = 0.95
    area_rollback = 0.9
    rollback_iter = 50
    radius = 500


    grabcut_iter = 5
    bubbles_size_factor = 0.3
    show_grabcut_results = True
    grabcut_extra = 50
    grabcut_rz_factor = 2
    grabcut_rz_threshold = 150

    log = True

def log_print(flag, string):
    if flag:
        print(string)

class VOTLT_Results_Saver(object):
    def __init__(self, save_path, video, t):
        result_path = os.path.join(save_path, 'rgbd-unsupervised')
        if not os.path.exists(os.path.join(result_path, video)):
            os.makedirs(os.path.join(result_path, video))
        self.g_region = open(os.path.join(result_path, video, video + '_001.txt'), 'w')
        self.g_region.writelines('1\n')
        self.g_conf = open(os.path.join(result_path, video, video + '_001_confidence.value'), 'w')
        self.g_conf.writelines('\n')
        self.g_time = open(os.path.join(result_path, video, video + '_001_time.value'), 'w')
        self.g_time.writelines([str(t)+'\n'])

    def record(self, conf, region, t):
        self.g_conf.writelines(["%f" % conf + '\n'])
        self.g_conf.flush()
        self.g_region.writelines(["%.4f" % float(region[0]) + ',' + "%.4f" % float(
            region[1]) + ',' + "%.4f" % float(region[2]) + ',' + "%.4f" % float(region[3]) + '\n'])
        self.g_region.flush()
        self.g_time.writelines([str(t)+'\n'])
        self.g_time.flush()

def get_seq_list(Dataset, mode=None, classes=None, video=None):
    if Dataset == "votlt18":
        data_dir = votlt18_dir
    elif Dataset == 'otb':
        data_dir = otb_dir
    elif Dataset == "votlt19":
        data_dir = votlt19_dir
    elif Dataset == "tlp":
        data_dir = tlp_dir
    elif Dataset == "lasot":
        data_dir = os.path.join(lasot_dir, classes)
    elif Dataset == 'demo':
        data_dir = '../demo_sequences'
    elif Dataset in ['cdtb_depth', 'cdtb_colormap', 'cdtb_color', 'cdtb_rgbd']:
        data_dir = cdtb_dir

    print(Dataset)
    sequence_list = os.listdir(data_dir)
    sequence_list.sort()
    sequence_list = [title for title in sequence_list if not title.endswith("txt")]
    if video is not None:
        sequence_list = [video]
    testing_set_dir = '../utils/testing_set.txt'
    testing_set = list(np.loadtxt(testing_set_dir, dtype=str))
    if mode == 'test' and Dataset == 'lasot':
        print('test data')
        sequence_list = [vid for vid in sequence_list if vid in testing_set]
    elif mode == 'train' and Dataset == 'lasot':
        print('train data')
        sequence_list = [vid for vid in sequence_list if vid not in testing_set]
    else:
        print("all data")

    return sequence_list, data_dir


def get_groundtruth(Dataset, data_dir, video):
    if Dataset == "votlt" or Dataset == "votlt19" or Dataset == "demo":
        sequence_dir = data_dir + '/' + video + '/color/'
        gt_dir = data_dir + '/' + video + '/groundtruth.txt'
    elif Dataset == "otb":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth_rect.txt'
    elif Dataset == "lasot":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth.txt'
    elif Dataset == "tlp":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth_rect.txt'
    elif Dataset in ['cdtb_depth', 'cdtb_colormap']:
        sequence_dir = data_dir + '/' + video + '/depth/'
        gt_dir = data_dir + '/' + video + '/groundtruth.txt'
    elif Dataset == 'cdtb_color':
        sequence_dir = data_dir + '/' + video + '/color/'
        gt_dir = data_dir + '/' + video + '/groundtruth.txt'
    elif Dataset == 'cdtb_rgbd':
        color_sequence_dir = data_dir + '/' + video + '/color/'
        depth_sequence_dir = data_dir + '/' + video + '/depth/'
        sequence_dir = {}
        sequence_dir['color'] = color_sequence_dir
        sequence_dir['depth'] = depth_sequence_dir
        gt_dir = data_dir + '/' + video + '/groundtruth.txt'

    try:
        groundtruth = np.loadtxt(gt_dir, delimiter=',')
    except:
        groundtruth = np.loadtxt(gt_dir)
    if Dataset == 'tlp':
        groundtruth = groundtruth[:, 1:5]

    return sequence_dir, groundtruth

def read_depth(file_name, depth_threshold=None, normalize=True):
    depth = cv2.imread(file_name, -1)
    if normalize:
        if depth_threshold is not None:
            try:
                max_depth = np.max(depth)
                if max_depth > depth_threshold:
                    depth[depth > depth_threshold] = depth_threshold
            except:
                depth = depth
        depth = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth = cv2.merge((depth, depth, depth)) # H * W * 3
    return depth # np.asarray(depth, dtype=np.uint8)

def read_colormap(file_name, depth_threshold=None):
    depth = cv2.imread(file_name, -1)
    if depth_threshold is not None:
        try:
            max_depth = np.max(depth)
            # print('max_depth : ', max_depth)
            if max_depth > depth_threshold:
                depth[depth > depth_threshold] = depth_threshold
        except:
            depth = depth
    depth = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth = np.asarray(depth, dtype=np.uint8)
    colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return colormap


def read_depth_and_target_depth(file_name, target_box=None, init=False, p=None, prev_depth=None):

    '''  1) get the depth image '''
    depth = cv2.imread(file_name, -1)
    depth = np.nan_to_num(depth, nan=np.max(depth))

    '''2) Decide the target depth '''
    if target_box is None:
        # print('Error : The target box or is required !!!')
        return depth, None

    target_box = [int(bb) for bb in target_box]
    target_depth, _ = get_target_depth(depth, target_box, p=p, init=init, last_depth=prev_depth)

    return depth, target_depth

def get_layered_image_by_depth(depth_image, target_depth, p):

    if target_depth is not None:
        low = max(target_depth-p.radius, 0)
        high = target_depth + p.radius

        layer = depth_image.copy()
        layer[layer < low] = high + 10
        layer[layer > high] = high + 10
    else:
        layer = depth_image.copy()

    layer = remove_bubbles(layer, bubbles_size=200)

    if p.dtype == 'centered_colormap':
        layer = cv2.normalize(layer, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        layer = np.asarray(layer, dtype=np.uint8)
        # layer = cv2.equalizeHist(layer)
        layer = cv2.applyColorMap(layer, cv2.COLORMAP_JET)

    return layer

def remove_bubbles(image, bubbles_size=100):
    try:
        binary_map = (image > 0).astype(np.uint8)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        mask = np.zeros((image.shape), dtype=np.uint8)
        for i in range(0, nb_components):
            if sizes[i] >= bubbles_size:
                mask[output == i+1] = 1

        if len(image.shape)>2:
            image = image * mask[:, :, np.newaxis]
        else:
            image = image * mask
    except:
        pass

    return image

def fill_holes(input_image):
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out


def get_target_depth(depth, target_box, p=None, init=False, last_depth=None):

    '''
        To estimate the target depth by using cv2.grabCut
    '''

    H, W = depth.shape

    target_box = [int(bb) for bb in target_box]
    x0, y0, w0, h0 = target_box
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x0+w0, W)
    y1 = min(y0+h0, H)
    possible_target = depth[y0:y1, x0:x1]
    # max_depth = np.max(possible_target)
    median_depth = np.median(possible_target) + 10

    bubbles_size = int(target_box[2]*target_box[3]*p.bubbles_size_factor * 0.25)
    log_print(p.log, 'bubbles_size in get_target_depth: %d '%bubbles_size)

    ''' add the surrounding extra pixels as the background '''
    extra_y0 = max(y0 - p.grabcut_extra, 0)
    extra_x0 = max(x0 - p.grabcut_extra, 0)
    extra_y1 = min(y1 + p.grabcut_extra, H)
    extra_x1 = min(x1 + p.grabcut_extra, W)

    rect_x0 = x0 - extra_x0
    rect_y0 = y0 - extra_y0
    rect_x1 = min(rect_x0 + w0, extra_x1)
    rect_y1 = min(rect_y0 + h0, extra_y1)
    rect = [rect_x0, rect_y0, rect_x1-rect_x0, rect_y1-rect_y0]

    target_patch = depth[extra_y0:extra_y1, extra_x0:extra_x1]
    target_patch = np.nan_to_num(target_patch, nan=np.max(target_patch))
    # target_patch = fill_holes(target_patch)

    image = target_patch.copy()
    image[image>median_depth*2] = median_depth*2 # !!!!!!!!!!
    image[image < 10] = median_depth*2

    i_H, i_W = image.shape

    '''To downsample the target_patch in order to speed up the cv2.grabCut'''
    rz_factor = p.grabcut_rz_factor if min(i_W, i_H) > p.grabcut_rz_threshold else 1
    if rz_factor > 1 :
        log_print(p.log, '!!!!!!!!!!!!! image is too large, to resize the image by factor %f'%rz_factor)
    rect_rz = [int(rt//rz_factor) for rt in rect]
    rz_dim = (int(i_W//rz_factor), int(i_H//rz_factor))
    image = cv2.resize(image, rz_dim, interpolation=cv2.INTER_AREA)
    rz_depth_patch =  cv2.resize(target_patch, rz_dim, interpolation=cv2.INTER_AREA)

    image = remove_bubbles(image, bubbles_size=bubbles_size)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = np.asarray(image, dtype=np.uint8)
    # image = cv2.equalizeHist(image)
    # image = cv2.merge((image, image, image))
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)


    mask = np.zeros(image.shape[:2], np.uint8)

    # if last_depth is not None:
    #     fore_indicator = rz_depth_patch.copy()
    #
    #     fore_indicator[fore_indicator < last_depth-50] = 0
    #     fore_indicator[fore_indicator > last_depth+50] = 0
    #     fore_indicator[fore_indicator > 0] = 3
    #     fore_indicator = np.asarray(fore_indicator, dtype=np.uint8)
    #
    #     mask = mask + fore_indicator

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    ''' 0-pixels and 2-pixels are background(set to 0), 1-pixels and 3-pixels are foreground(set to 1)'''
    log_print(p.log, 'starting grabCut ...')
    grabCut_tic = time.time()
    cv2.grabCut(image, mask, rect_rz, bgdModel, fgdModel, p.grabcut_iter, cv2.GC_INIT_WITH_RECT) ##  cv2.GC_INIT_WITH_RECT) #
    log_print(p.log, 'grabCut used time : %f'%(time.time() - grabCut_tic))

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask2 = remove_bubbles(mask2, bubbles_size=bubbles_size)


    ''' Resize back to original size '''
    image = cv2.resize(image, (i_W, i_H), interpolation=cv2.INTER_AREA)
    mask2 = cv2.resize(mask2, (i_W, i_H), interpolation=cv2.INTER_AREA)


    ''' get the new optimal rect box '''
    #
    row_sum = np.sum(mask2, axis=1)
    col_sum = np.sum(mask2, axis=0)
    row_nonzero = np.nonzero(row_sum)[0]
    col_nonzero = np.nonzero(col_sum)[0]

    if len(row_nonzero)>0:
        new_y0 = row_nonzero[0]
        new_y1 = row_nonzero[-1]
    else:
        new_y0 = rect[1]
        new_y1 = rect[1]+rect[3]
    if len(col_nonzero) > 0:
        new_x0 = col_nonzero[0]
        new_x1 = col_nonzero[-1]
    else:
        new_x0 = rect[0]
        new_x1 = rect[0]+rect[2]
    # convert new_rect back to original image coordinates
    new_rect = [new_x0+extra_x0, new_y0+extra_y0, new_x1-new_x0+1, new_y1-new_y0+1]

    if new_rect[2]*rect[3] < target_box[2]*target_box[3]*0.5:
        new_rect = target_box

    ''' to get the target depth values '''
    target_pixels = target_patch * mask2
    target_pixels = target_pixels.flatten()
    target_pixels.sort()
    target_pixels = target_pixels[target_pixels>0]

    if len(target_pixels) > p.minimun_target_pixels:
        # target_depth = np.median(target_pixels) # target_depth = target_pixels[int(len(target_pixels)/2)]
        #
        hist, bin_edges = np.histogram(target_pixels, bins=20)
        peak_idx = np.argmax(hist)
        selected_target_pixels = target_pixels
        target_depth_low = bin_edges[peak_idx]
        target_depth_high= bin_edges[peak_idx+1]
        selected_target_pixels = selected_target_pixels[selected_target_pixels<=target_depth_high]
        selected_target_pixels = selected_target_pixels[selected_target_pixels>=target_depth_low]
        target_depth = np.median(selected_target_pixels)

        print('max and min seletected_target_pixels: ', np.min(selected_target_pixels), np.max(selected_target_pixels))
        log_print(p.log, 'Target_depth + %f,  Range %f  -  %f'%(target_depth, target_depth_low,target_depth_high))

    else:
        log_print(p.log, 'can not find the target depth, use the mean value of bbox')
        target_depth = median_depth

    if p.grabcut_visualization:
        cv2.namedWindow('grabcut',cv2.WINDOW_NORMAL)

        cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), [255, 255, 255], 2)  # [0, 102, 51]

        # stack_mask01 = np.asarray(mask*255, np.uint8)
        mask = cv2.resize(mask, (i_W, i_H), interpolation=cv2.INTER_AREA)
        stack_mask01 = np.asarray(mask/4*255, np.uint8)
        stack_mask01 = cv2.merge((stack_mask01, stack_mask01, stack_mask01))
        cv2.rectangle(stack_mask01, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), [255, 255, 255], 2)  # [0, 102, 51]
        # cv2.rectangle(stack_mask01, (new_x0, new_y0), (new_x1, new_y1), [255, 0, 0], 2)  # [0, 102, 51]


        stack_mask = np.asarray(mask2*255, np.uint8)
        stack_mask = cv2.merge((stack_mask, stack_mask, stack_mask))
        cv2.rectangle(stack_mask, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), [0, 0, 255], 2)  # [0, 102, 51]
        cv2.rectangle(stack_mask, (new_x0, new_y0), (new_x1, new_y1), [255, 0, 0], 2)  # [0, 102, 51]

        foreground_colorm = image * mask2[:, :, np.newaxis]
        cv2.rectangle(foreground_colorm, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), [255, 255, 255], 2)  # [0, 102, 51]

        show_imgs = cv2.hconcat((image, stack_mask01, stack_mask, foreground_colorm))
        cv2.imshow('grabcut', show_imgs)
        cv2.waitKey(1)



        depth = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        depth = np.asarray(depth, dtype=np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
        cv2.imshow('depth', depth)
        cv2.waitKey(1)


        log_print(p.log, 'Estimated Traget depth = %f'%target_depth)

    return target_depth, new_rect

def read_rgbd(file_name, normalize=True):
    '''
    cv2.imread() return uint8 -> float
    depth is float32
    '''
    color_filename = file_name['color']
    depth_filename = file_name['depth']

    depth = cv2.imread(depth_filename, -1)
    if normalize:
        depth = cv2.normalize(dpeth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth = np.float32(depth)

    color = np.asarray(cv2.imread(color_filename), dtype=np.float32)            # H * W * 3
    b, g, r = color[..., 0], color[..., 1], color[..., 2]
    rgbd = cv2.merge((r, g, b, depth))
    return rgbd # np.asarray(rgbd, dtype=np.float32)
    # Song : remember to check the value in depth change or not , from uint8 to 32f ???

def get_image(image_dir, target_box=None, init=False, p=None, prev_depth=None):
    '''
    Parameters for the layered inputs , e.g. dtype= layered_colormap, layered_raw_depth, layered_normalized_depth
        -     K      : the number of layers to be splitted
        - layer_bin  : the distance of each layer
        - layer_index: the index of the layer containg the target
        - target_box : the gt box or the predicted box containing the possible target
    '''
    if p.dtype == 'rgb':
        image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
    elif p.dtype == 'raw_depth':
        image = read_depth(image_dir, normalize=False) # H * W * 3, [dp, dp, dp]
    elif p.dtype == 'normalized_depth':
        image = read_depth(image_dir, normalize=True) # H * W * 3, [dp, dp, dp] normalized
    elif p.dtype == 'colormap':
        image = read_colormap(image_dir, depth_threshold=depth_threshold) # H * W * 3
    elif p.dtype == 'raw_rgbd':
        image = read_rgbd(image_dir, normalize=False) # H * W * 4, rgb + raw_depth
    elif p.dtype == 'normalized_rgbd':
        image = read_rgbd(image_dir, normalize=True) # H * W * 4, rgb + normalized_depth

    elif p.dtype in ['centered_colormap', 'centered_raw_depth', 'centered_normalized_depth']:
        ''' get the depth image and possible target depth '''
        image = read_depth_and_target_depth(image_dir, target_box=target_box, init=init, p=p, prev_depth=prev_depth)
    else:
        print('unknown input type : %s'%dtype)
        image = None

    return image

def get_averageArea_from_history(scores, bboxes, index, N=10):
    ''' get the average area of previous N reliable (conf > 0.5) frames '''
    avgArea = 0.0
    count = 0
    while count < N and index > -1:
        if scores[index] < 0.5:
            index -= 1
        else:
            box = bboxes[index, :]
            avgArea += box[2]*box[3]
            count += 1
            index -= 1

    return 1.0 * avgArea / count if count > 0 else 0.0

def get_averageDepth_from_history(scores, depths, index, N=10):
    ''' get the average area of previous N reliable (conf > 0.5) frames '''
    avgDepth = 0.0
    count = 0
    while count < N and index > -1:
        if scores[index] < 0.5:
            index -= 1
        else:
            avgDepth += depths[index]
            count += 1
            index -= 1

    return 1.0 * avgDepth / count if count > 0 else 0.0

def get_history_depths(scores, depths, index, N=10):
    ''' get the average area of previous N reliable (conf > 0.5) frames '''
    depth_list = []
    count = 0
    while count < N and index > -1:
        if scores[index] < 0.5:
            index -= 1
        else:
            depth_list.append(depths[index])
            count += 1
            index -= 1

    return depth_list

def get_averageScore_from_history(scores, index, N=10):
    ''' get the average area of previous N frames '''
    avgScore = 0.0
    count = 0
    while count < N and index > -1:
        avgScore += scores[index]
        count += 1
        index -= 1

    return 1.0 * avgScore / count if count > 0 else 0.0

# def get_layer_from_images(image, layer_index=None, layer_depth=None, dtype='centered_colormap', bubbles_size=150, radius=1000):
#
#     if layer_index is None and layer_depth is None:
#         print('Error !!!!! There must be layer_index or layer_depth')
#         return None
#     if layer_index is not None:
#         temp_image = image[..., layer_index]
#     else:
#         temp_image = get_layered_image_by_depth(image[..., 1], layer_depth, p)
#
#     if dtype == 'layered_colormap' or dtype == 'centered_colormap':
#         temp_image = cv2.normalize(temp_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         temp_image = np.asarray(temp_image, dtype=np.uint8)
#         temp_image = cv2.applyColorMap(temp_image, cv2.COLORMAP_JET)
#         temp_image = remove_bubbles(temp_image, bubbles_size=bubbles_size)
#
#     elif dtype == 'layered_raw_depth' or dtype == 'centered_raw_depth':
#         temp_image = temp_image
#         temp_image = cv2.merge((temp_image, temp_image, temp_image))
#     elif dtype == 'layered_normalized_depth' or dtype == 'centered_normalized_depth':
#         temp_image = cv2.normalize(temp_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         temp_image = np.asarray(temp_image, dtype=np.uint8)
#         temp_image = cv2.merge((temp_image, temp_image, temp_image))
#     else:
#         print("Wrong dtype in get_layer_from_images")
#         temp_image = None
#     return temp_image

def search_layer(tracker, layer_depth, image, p, prev_avgArea, prev_avgDepth, prev_center, init_area, temp_results,
                 last_gt=None,
                 use_conf=True, use_max_min_scale=True, use_init_scale=True, use_depth_scale=True, use_center_scale=True,
                 ):
# def search_layer(tracker, layer_index, layer_depth, image, p, prev_avgArea, prev_avgDepth, prev_center, init_area, temp_results,
#                  use_conf=True, use_max_min_scale=True, use_init_scale=True, use_depth_scale=True, use_center_scale=True,
#                  ):

    # temp_region, temp_score_map, temp_iou, temp_score_max, temp_dis, temp_target_depth = temp_results
    ''' search in previous or next layer'''
    # input_image = get_layer_from_images(image, layer_index=layer_index, layer_depth=layer_depth, dtype=p.dtype, radius=p.radius, bubbles_size=int(init_area*p.bubbles_size_factor))
    input_image = get_layered_image_by_depth(image, layer_depth, p)

    temp_results_in_layer = tracker.tracking(input_image, count=False, manually_update=False, last_gt=last_gt)

    # update = True
    update = 0

    ''' Consider the confidence '''
    if use_conf:
        # conf_changes_in_layer = temp_results_in_layer[-2] > max(temp_results[-3], p.conf_threshold)
        conf_changes_in_layer = temp_results_in_layer[-2] > temp_results[-3]
        # update = update and  conf_changes_in_layer
        if conf_changes_in_layer:
            update += 1

    '''Consider the Area or Shape change'''
    temp_region_in_layer = temp_results_in_layer[0]
    temp_area_in_layer = temp_region_in_layer[2] * temp_region_in_layer[3]

    ''' x-y location changes '''
    temp_center_in_layer = [int(temp_region_in_layer[0]+temp_region_in_layer[2]/2), int(temp_region_in_layer[1]+temp_region_in_layer[3]/2)]

    ''' depth changes'''
    target_depth_in_layer, finetuned_region = get_target_depth(image, temp_region_in_layer, p=p, last_depth=prev_avgDepth)
    log_print(p.log, 'target_depth_in_layer: %f'%target_depth_in_layer)

    if use_max_min_scale:
        area_changes_in_layer = (1.0 * max(temp_area_in_layer, prev_avgArea) / (min(temp_area_in_layer, prev_avgArea)+1.0)) < p.area_scale_threshold
        # update = update and area_changes_in_layer
        if area_changes_in_layer:
            update += 1

    if use_init_scale:
        area_minimum_in_layer = temp_area_in_layer > init_area *  p.minimun_area_threshold
        # update = update and area_minimum_in_layer
        update += int(area_minimum_in_layer=='true')

    if use_center_scale:
        center_change_in_layer = np.linalg.norm(np.asarray(temp_center_in_layer) - np.asarray(prev_center)) < np.sqrt(prev_avgArea)*1.5
        # update = update and center_change_in_layer

        if center_change_in_layer:
            update += 1

    ''' Consider the depth changes '''
    if use_depth_scale:
        # target_depth_changes_in_layer = (1.0 * max(target_depth_in_layer, prev_avgDepth) / (min(target_depth_in_layer, prev_avgDepth)+1.0))  < p.target_depth_changes_threshold
        target_depth_changes_in_layer = abs(target_depth_in_layer - prev_avgDepth) > 500
        # update = update and target_depth_changes_in_layer
        if target_depth_changes_in_layer:
            update += 1


    # if update:
    if update > 3:
        temp_region, score_map, temp_iou, temp_score_max, temp_dis  = temp_results_in_layer
        temp_results = (finetuned_region, score_map, temp_iou, temp_score_max, temp_dis, target_depth_in_layer)
        tracker.manually_update(input_image)

    print('- Move: ', update>3, ' layer_depth: ', layer_depth, ' score: ', temp_results_in_layer[-2], 'area : ', temp_area_in_layer, 'center: ', temp_center_in_layer, 'init_area: ', init_area)

    return temp_results, update>2

def run_seq_list(Dataset, p, sequence_list, data_dir, tracker_name='dimp', tracker_params='dimp50'):
    '''Song's comments
        - p.dtype : the data type for inputs, e.g. rgb, colormap, depth, rgbd
            - rgb      : H x W x 3
            - depth    : 1) normalzie to [0, 255], 2) concatenate the channels , [depth, depth, depth]
                - raw depth
                - normalized_depth
            - colormap : 1) normalzie depth images, 2) convert to colormap, JET
            - rgbd     : H x W x (3+1), rgb+depth, [..., :3] = rgb, [..., 3:] = depth

            - layered_colormap : H x W x K , K is the number of layers, we only apply the layer-based search strategy
            - layered_raw_depth    : H x W x K , K is the number of layers
            - layered_normalized_depth

            - centered_colormap
            - centered_raw_depth
            - centered_normalized_depth
    '''

    m_shape = 19
    base_save_path = os.path.join('/home/yan/Data2/vot-workspace/results/', p.name, Dataset+'+'+p.dtype)

    for seq_id, video in enumerate(sequence_list):
        sequence_dir, groundtruth = get_groundtruth(Dataset, data_dir, video)

        if p.save_training_data:
            result_save_path = os.path.join(base_save_path, 'train_data', video + '.txt')
            if os.path.exists(result_save_path):
                continue
        if p.save_results:
            result_save_path = os.path.join(base_save_path, 'rgbd-unsupervised', video, video + '_001.txt')
            if os.path.exists(result_save_path):
                continue

        if p.dtype == 'rgbd':
            color_images = os.listdir(sequence_dir['color'])
            color_images.sort()
            image_list = [im[:-4] for im in color_images if im.endswith("jpg") or im.endswith("jpeg") or im.endswith("png")]
            image_dir = {}
            image_dir['color'] = sequence_dir['color'] + image_list[0] + '.jpg'
            image_dir['depth'] = sequence_dir['depth'] + image_list[0] + '.png'
        else:
            image_list = os.listdir(sequence_dir)
            image_list.sort()
            image_list = [im for im in image_list if im.endswith("jpg") or im.endswith("jpeg") or im.endswith("png")]
            image_dir = sequence_dir + image_list[0]

        region = Region(groundtruth[0, 0], groundtruth[0, 1], groundtruth[0, 2], groundtruth[0, 3])
        region1 = groundtruth[0]

        if p.dtype in ['centered_colormap', 'centered_raw_depth', 'centered_normalized_depth']:
            image, init_target_depth = get_image(image_dir, target_box=region1, init=True, p=p, prev_depth=None)
        else:
            ''' return H*W*3 images, e.g. colormap, rgb, raw_depth, normalized depth , Or rgbd'''
            image = get_image(image_dir, dtype=p.dtype, depth_threshold=p.depth_threshold)

        h = image.shape[0]
        w = image.shape[1]

        box = np.array([region1[0] / w, region1[1] / h, (region1[0] + region1[2]) / w, (region1[1] + region1[3]) / h]) # w, h in (0 , 1)
        tic = time.time()

        if p.dtype in ['centered_colormap', 'centered_raw_depth', 'centered_normalized_depth']:
            # input_image = get_layer_from_images(image, 0, p.dtype)
            input_image = get_layered_image_by_depth(image, init_target_depth, p)
        else:
            input_image = image

        tracker = Dimp_LTMU_Tracker(input_image, region, p=p, groundtruth=groundtruth, tracker_name=tracker_name, tracker_params=tracker_params)

        ''' -------------- Song : we add some new parameters here ------------'''
        ''' Song : we keep the initial tracker for reset '''
        init_tracker = tracker
        reset = False
        ''' Song : we keep the init area of the target, if the current area < init_area*0.1 ? '''
        init_area = region1[2] * region1[3]
        init_center = [int(region1[0] + region1[2]/2), int(region1[1]+region1[3]/2)]
        ''' ------------------------------------------------------------------ '''

        score_map, score_max = tracker.get_first_state()
        t = time.time() - tic
        if p.save_results and Dataset in ['votlt18', 'votlt19', 'cdtb_colormap', 'cdtb_color', 'cdtb_depth', 'cdtb_rgbd']:
            results_saver = VOTLT_Results_Saver(base_save_path, video, t)
        num_frames = len(image_list)
        all_map = np.zeros((num_frames, m_shape, m_shape))
        all_map[0] = cv2.resize(score_map, (m_shape, m_shape))
        bBoxes_results = np.zeros((num_frames, 4))
        bBoxes_results[0, :] = region1
        bBoxes_train = np.zeros((num_frames, 8))
        bBoxes_train[0, :] = [box[0], box[1], box[2], box[3], 0, 1, score_max, 0]

        # Song : Record the layer index and the reliable score
        if p.dtype in ['centered_colormap', 'centered_raw_depth', 'centered_normalized_depth']:
            previous_depths = np.zeros((num_frames,))
            previous_depths[0] = init_target_depth

            previous_scores = np.zeros((num_frames,))
            previous_scores[0] = 1.0

            previous_centers = dict()
            previous_centers[0] = init_center

            previous_areas = np.zeros((num_frames, ))
            previous_areas[0] = init_area

        tracker_copy = init_tracker

        for im_id in range(1, len(image_list)):
            tic = time.time()

            if p.dtype == 'rgbd':
                image_dir = {}
                image_dir['color'] = sequence_dir['color'] + image_list[im_id] + '.jpg'
                image_dir['depth'] = sequence_dir['depth'] + image_list[im_id] + '.png'
            else:
                image_dir = sequence_dir + image_list[im_id]

            if p.dtype in ['centered_colormap', 'centered_raw_depth', 'centered_normalized_depth']:

                if not reset:
                    ''' get a reliable target_depth and score , whose score > p.conf_threshold '''
                    temp_im_id = im_id
                    prev_depth = previous_depths[temp_im_id-1]
                    prev_score = previous_scores[temp_im_id-1]
                    prev_center = previous_centers[temp_im_id-1]
                    prev_area = previous_areas[temp_im_id-1]
                    prev_bbox = bBoxes_results[temp_im_id - 1, :]

                    while prev_score < p.conf_threshold and temp_im_id > 1:
                        temp_im_id -= 1
                        prev_depth = previous_depths[temp_im_id-1]
                        prev_score = previous_scores[temp_im_id-1]
                        prev_center = previous_centers[temp_im_id-1]
                        prev_area = previous_areas[temp_im_id-1]
                        prev_bbox = bBoxes_results[temp_im_id - 1, :]

                    n_history = 20
                    prev_avgScore = get_averageScore_from_history(previous_scores, im_id-1, N=n_history)
                    prev_avgArea  = get_averageArea_from_history(previous_scores, bBoxes_results, im_id-1, N=n_history)
                    prev_avgDepth = get_averageDepth_from_history(previous_scores, previous_depths, im_id-1, N=n_history)
                else:
                    ''' Re-set the tracker , using the inital parameters '''
                    tracker = init_tracker
                    prev_depth = init_target_depth
                    prev_avgDepth = init_target_depth
                    prev_avgArea = init_area
                    prev_area = init_area
                    prev_bbox = region1
                    prev_avgScore = 1.0
                    reset = False

                image, _, = get_image(image_dir, target_box=None, init=False, p=p, prev_depth=prev_depth)
            else:
                image = get_image(image_dir, dtype=p.dtype, depth_threshold=p.depth_threshold)

            ''' Tracking process '''
            if p.dtype in ['centered_colormap', 'centered_raw_depth', 'centered_normalized_depth']:

                ''' 1) search the target in the optimal layer '''
                # bubbles_size = int(init_area*0.4)
                # print('bubbles_size: ', bubbles_size)


                # input_image = get_layer_from_images(image, 0, p.dtype, bubbles_size=bubbles_size)

                input_image = get_layered_image_by_depth(image, prev_depth,  p)

                centered_temp_region, centered_score_map, centered_temp_iou, centered_temp_score_max, centered_temp_dis = tracker.tracking(input_image, manually_update=False, last_gt=prev_bbox)
                # temp_results = (centered_temp_region, centered_score_map, centered_temp_iou, centered_temp_score_max, centered_temp_dis)

                ''' Compare the predicted depth to the previous history depths ? or the previous depth ????'''
                temp_area = centered_temp_region[2] * centered_temp_region[3]

                temp_target_depth, finetuned_region = get_target_depth(image, centered_temp_region, p=p, last_depth=prev_depth)
                temp_center = [int(finetuned_region[0]+finetuned_region[2]/2), int(finetuned_region[1]+finetuned_region[3]/2)]

                ''' using the finetuned region and the targetdepth '''
                temp_results = (finetuned_region, centered_score_map, centered_temp_iou, centered_temp_score_max, centered_temp_dis, temp_target_depth)

                '''
                How to decide that prediction is not reliable ???

                    - conf  > conf_threshold
                    - area  vs pre_area   or History area ??
                    - depth vs prev_depth or History depth ??
                    - x-y position vs prev_center


                How to decide the searching layers ?
                    - whole depth image
                    - move the center , e.g. new_center = center +/- radius
                '''
                # temp_depth_changes_flag = 1.0 * max(temp_target_depth, prev_depth) / (min(temp_target_depth, prev_depth)+1.0) > p.target_depth_changes_threshold
                # temp_depth_changes_flag = 1.0 * max(temp_target_depth, prev_avgDepth) / (min(temp_target_depth, prev_avgDepth)+1.0) > p.target_depth_changes_threshold
                # temp_depth_changes_flag = 1.0 * max(temp_target_depth, prev_depth) / (min(temp_target_depth, prev_depth)+1.0) > p.target_depth_changes_threshold
                temp_depth_changes_flag = abs(temp_target_depth - prev_depth) > 500
                # temp_area_changes_flag = temp_area < prev_avgArea * p.minimun_area_threshold
                temp_area_changes_flag = temp_area < prev_area * p.minimun_area_threshold

                temp_conf_changes_flag = centered_temp_score_max < p.conf_threshold
                # temp_center_changes_flag = np.linalg.norm(np.asarray(temp_center) - np.asarray(prev_center)) > np.sqrt(temp_area)*1.2
                # temp_center_changes_flag = np.linalg.norm(np.asarray(temp_center) - np.asarray(prev_center)) > np.sqrt(temp_area)*0.5
                temp_center_changes_flag = np.linalg.norm(np.asarray(temp_center) - np.asarray(prev_center)) > np.sqrt(prev_area)*1.5

                if temp_conf_changes_flag or temp_area_changes_flag or temp_depth_changes_flag or temp_center_changes_flag:

                    update = False

                    if not update:
                        ''' search the whole image with current tracker '''
                        # temp_results, update = search_layer(tracker, None, image, p, prev_avgArea, prev_avgDepth, prev_center, init_area, temp_results)
                        temp_results, update = search_layer(tracker, None, image, p, prev_area, prev_depth, prev_center, init_area, temp_results, last_gt=prev_bbox)


                    if not update:
                        ''' search the whole image with init tracker '''
                        # temp_results, update = search_layer(init_tracker, None, image, p, prev_avgArea, prev_avgDepth, prev_center, init_area, temp_results)
                        temp_results, update = search_layer(init_tracker, None, image, p, prev_area, prev_depth, prev_center, init_area, temp_results, last_gt=prev_bbox)

                        if update:
                            tracker = init_tracker

                    if not update:
                        # layer_depth = prev_depth
                        depth_list = get_history_depths(previous_scores, previous_depths, im_id, N=2)
                        for layer_depth in depth_list:
                            # temp_results, update = search_layer(tracker, layer_depth, image, p, prev_avgArea, prev_avgDepth, prev_center, init_area, temp_results)
                            temp_results, update = search_layer(tracker, layer_depth, image, p, prev_area, prev_depth, prev_center, init_area, temp_results, last_gt=prev_bbox)
                            if update:
                                break
                    if not update:
                        # layer_depth = prev_depth
                        depth_list = get_history_depths(previous_scores, previous_depths, im_id, N=2)
                        for layer_depth in depth_list:
                            # temp_results, update = search_layer(init_tracker, layer_depth, image, p, prev_avgArea, prev_avgDepth, prev_center, init_area, temp_results)
                            temp_results, update = search_layer(init_tracker, layer_depth, image, p, prev_area, prev_depth, prev_center, init_area, temp_results, last_gt=prev_bbox)

                            if update:
                                tracker = init_tracker
                                break

                    ''' search different layers starting from current depth center '''

                    search_steps = [-0.5 * p.radius, 0.5 * p.radius]
                    # search_steps = [-0.5 * p.radius, 0.5 * p.radius]


                    for step in search_steps:
                        layer_depth = temp_target_depth + step
                        if layer_depth > 0 and layer_depth < 2 * temp_target_depth:
                            ''' use the current tracker '''
                            # temp_results, update = search_layer(tracker, layer_depth, image, p, prev_avgArea, prev_avgDepth, prev_center, init_area, temp_results)
                            temp_results, update = search_layer(tracker, layer_depth, image, p, prev_area, prev_depth, prev_center, init_area, temp_results, last_gt=prev_bbox)
                            if update:
                                break
                            ''' use the init tracker '''
                            # temp_results, update = search_layer(init_tracker, layer_depth, image, p, prev_avgArea, prev_avgDepth, prev_center, init_area, temp_results)
                            temp_results, update = search_layer(init_tracker, layer_depth, image, p, prev_area, prev_depth, prev_center, init_area, temp_results, last_gt=prev_bbox)
                            if update:
                                tracker = init_tracker
                                break

                # else:
                #     print('!!!!!!!!!!!!!!!!!!!! update trackers !!!!!!!!!!!!!!!!')
                #     tracker.manually_update(input_image)

                if im_id % 50 == 0:
                    temp_results, update = search_layer(init_tracker, prev_depth, image, p, prev_area, prev_depth, prev_center, init_area, temp_results, last_gt=prev_bbox)
                    if update:
                        tracker = init_tracker
                    else:
                        temp_results, update = search_layer(init_tracker, None, image, p, prev_area, prev_depth, prev_center, init_area, temp_results, last_gt=prev_bbox)
                        if update:
                            tracker = init_tracker

                region, score_map, iou, score_max, dis, target_depth = temp_results

                temp_area = region[2]*region[3]

                previous_centers[im_id] = [int(region[0]+region[2]/2), int(region[1]+region[3]/2)]
                previous_depths[im_id] = target_depth
                previous_scores[im_id] = score_max
                previous_areas[im_id] = temp_area

                ''' If target moves too far '''
                # if  1.0 * max(target_depth, prev_depth) / (min(target_depth, prev_depth)+1.0)  > p.target_depth_changes_threshold:
                if abs(target_depth - prev_depth) > 500:
                    print('!!!!!!!!!!!!! Target moving in Depth too fast !!!!!!!!!!!!!!')
                    # print('Target depth : %f,   prev_depth : %f, ratio %f > %f'%(target_depth, prev_depth, 1.0 * max(target_depth, prev_depth) / (min(target_depth, prev_depth)+1.0), p.target_depth_changes_threshold))
                    print('Target depth : %f,   prev_depth : %f, T-P = %f'%(target_depth, prev_depth, abs(target_depth - prev_depth)))

                    # score_max = 0.1 * score_max
                    score_max = min(p.conf_threshold - 0.01, score_max)
                    previous_depths[im_id] = previous_depths[im_id-1]

                ''' if the area change too much, discard..., update the score '''
                # if temp_area < init_area * p.minimun_area_threshold:
                if temp_area < prev_area * p.minimun_area_threshold:
                    print('!!!!!!!!!!!!! Frame %d : the predicted area is too small  %f vs init_area %f!!!!!!!!!!!!!!!'%(im_id, temp_area, init_area))
                    # score_max = 1.0 * temp_area / prev_area
                    score_max = min(p.conf_threshold - 0.01, score_max)
                    previous_areas[im_id] = previous_areas[im_id-1]
                    print('temp_area : %f , prev_area, %f , prev_area * ration : %f'%(temp_area, prev_area, prev_area*p.minimun_area_threshold))

                ''' the normal distance between to bbox'''
                norm_dist = np.linalg.norm(np.asarray(bBoxes_results[im_id-1, :]) - np.asarray(region))
                print('norm_dist : %f' % norm_dist)


                moving_dist = np.linalg.norm(np.asarray(previous_centers[im_id]) - np.asarray(previous_centers[im_id-1]))
                dist_threshold = np.sqrt(prev_area)*2
                if moving_dist >  dist_threshold:
                    print('!!!!!!!!!!!!!! estimated depth too far away!!!!!!!!!!!!!!!!!')
                    # score_max = score_max * 0.1
                    score_max = min(p.conf_threshold - 0.01, score_max)
                    previous_centers[im_id] = previous_centers[im_id-1]
                    region = bBoxes_results[im_id-1, :]

                    print('previous_center[im_id-1] = ', previous_centers[im_id-1],
                          '  previous_center[in_id] = , ', previous_centers[im_id],
                          ' moving_dist: %f > %s '%(moving_dist, dist_threshold))


                print("%d: " % seq_id + video
                      + ": %d /" % im_id + "%d" % len(image_list)
                      + ' estimated target_depth / prev_depth : %f / %f'%(target_depth, prev_depth)
                      + ' estimated score: %f'%score_max
                      + ' estiamted area: %f / %f'%(temp_area, prev_area)
                      + ' moving_dist / dist_threshold : %f / %f'%(moving_dist, dist_threshold)
                      )
            else:
                region, score_map, iou, score_max, dis = tracker.tracking(image)
                print("%d: " % seq_id + video + ": %d /" % im_id + "%d" % len(image_list) + "score: %f"%score_max)

            if score_max < 0.2:
                tracker = tracker_copy

            if score_max > 0.95:
                tracker_copy = tracker

            if score_max > 0.8:
                print('!!!!!!!!!!!!!!!!!!!! update trackers !!!!!!!!!!!!!!!!')
                tracker.manually_update(input_image)

            t = time.time() - tic
            print('total time: ', t)
            if p.save_results and Dataset in ['votlt18', 'votlt19', 'cdtb_colormap', 'cdtb_color', 'cdtb_depth', 'cdtb_rgbd']:
                results_saver.record(conf=score_max, region=region, t=t)
            all_map[im_id] = cv2.resize(score_map, (m_shape, m_shape))

            box = np.array(
                [region[0] / w, region[1] / h, (region[0] + region[2]) / w, (region[1] + region[3]) / h])
            bBoxes_train[im_id, :] = [box[0], box[1], box[2], box[3], im_id, iou, score_max, dis]
            bBoxes_results[im_id, :] = region

        if p.save_training_data:
            np.savetxt(os.path.join(base_save_path, 'train_data', video + '.txt'), bBoxes_train,
                       fmt="%.8f,%.8f,%.8f,%.8f,%d,%.8f,%.8f,%.8f")
            np.save(os.path.join(base_save_path, 'train_data', video + '_map'), all_map)

        tracker.sess.close()
        tf.reset_default_graph()


def eval_tracking(Dataset, p, mode=None, video=None, tracker_name='dimp', tracker_params='dimp50'):

    if Dataset == 'lasot':
        classes = os.listdir(lasot_dir)
        classes.sort()
        for c in classes:
            sequence_list, data_dir = get_seq_list(Dataset, mode=mode, classes=c)
            run_seq_list(Dataset, p, sequence_list, data_dir, tracker_name=tracker_name, tracker_params=tracker_params)
    elif Dataset in ['votlt18', 'votlt19', 'tlp', 'otb', 'demo', 'cdtb_depth',  'cdtb_colormap', 'cdtb_color', 'cdtb_rgbd']:
        sequence_list, data_dir = get_seq_list(Dataset, video=video)
        run_seq_list(Dataset, p, sequence_list, data_dir, tracker_name=tracker_name, tracker_params=tracker_params)
    else:
        print('Warning: Unknown dataset.')
