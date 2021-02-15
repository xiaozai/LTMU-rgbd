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
        self.g_region.writelines(["%.4f" % float(region[0]) + ',' + "%.4f" % float(
            region[1]) + ',' + "%.4f" % float(region[2]) + ',' + "%.4f" % float(region[3]) + '\n'])
        self.g_time.writelines([str(t)+'\n'])


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

# def read_fixed_layered_depth(file_name, target_box=None, depth_threshold=None,
#                              K=10, layer_bin=1000, layer_index=None):
#     '''
#      To split the depth values :
#         - 1) fixed  bins, e.g. 0-1000, 1000-2000, 2000-3000, 3000-4000, ....
#         - 2) k-clustered based bins, e.g. by performing the K-clusters, to get the depth seeds, each seed is the center of layers
#                                      e.g. K-Cluster(depth) -> dp = 30, 1000, 4500, 8000 -> Layer_30, Layer_1000, .... Layer_dp
#
#     We use the 1) now !!!!!!, one problem, K and Step affect the performance, especially for indoor scene, needs the optimal K and Step
#
#     How to decide the K and layer_bin ??
#     1) according to the distance in the whole image ? closed (0-3m), medium distance (3m - 10m) , far (>10m)
#     2) according to the distance in the inital box ?? assum that the farset distance is 2*D_initbox ? then K = 10
#
#     '''
#     # 1) get the depth image
#     depth = cv2.imread(file_name, -1)
#
#     # 2) decide the K and layer bin and the target depth
#     if target_box is not None:
#         '''
#         Find the possible target depth, by using the K-means,
#             - assume that the init_box covers most the target, + few pixels belong to background(very large)
#             - num_cluster = 2, target + background
#             - calculate the most frequency pixels, which belong to target
#             - if there is only the target in the box, then two centers will be similar, it is okay!
#         '''
#         target_box = [int(bb) for bb in target_box]
#         target_patch = depth[target_box[1]:target_box[1]+target_box[3], target_box[0]:target_box[0]+target_box[2]]
#         target_depth_values = target_patch.copy().reshape((-1, 1))
#         num_cluster = 2
#         kmeans = KMeans(init="random", n_clusters=num_cluster, n_init=10, max_iter=300, random_state=42)
#         kmeans.fit(target_depth_values)
#         depth_centers = kmeans.cluster_centers_
#         depth_labels = kmeans.labels_
#         depth_frequencies = []
#         for ii in range(num_cluster):
#             depth_frequencies.append(np.sum(depth_labels[depth_labels==ii]))
#         target_depth = depth_centers[np.argmax(depth_frequencies)][0]
#
#     if layer_bin is None and K is None:
#
#         ''' according to the distance of the target '''
#         # max_depth = 2.0*target_depth
#         # K = 10
#         # layer_bin = max_depth // K
#
#         ''' according to the distance of the whole image'''
#         # Only for Initial images, roughly decide is indoor or outdoor
#         max_depth = np.max(depth)
#
#         if max_depth > 10000: # 12 m
#             layer_bin = 1000
#             K = 10
#         elif max_depth < 3000:
#             K = 5
#             layer_bin = max_depth // K
#         else:
#             K = 10
#             layer_bin = max_depth // K
#
#     if depth_threshold is not None:
#         try:
#             max_depth = np.max(depth)
#             if max_depth > depth_threshold:
#                 depth[depth > depth_threshold] = depth_threshold
#         except:
#             depth = depth
#
#     H, W = depth.shape[0], depth.shape[1]
#     depth_layers = np.zeros((H, W, K), dtype=np.float32)
#     target_in_layer = np.zeros((K,), dtype=np.int32) # count how many pixels of target patch in each layer
#
#     ''' Current we only conside the neighbour layers '''
#     if layer_index is not None:
#         neighbour_layers = 2                         # centered at layer_index, C-N : C+N
#         start_idx = layer_index - neighbour_layers   # layer_index-1 if layer_index-1 > 0 else 0
#         end_idx = layer_index + neighbour_layers+1   # layer_index+2 if layer_index+2 < K else K
#     else:
#         start_idx = 0
#         end_idx = K
#
#     ''' To get the layers
#         1) to decide the boarders, low and high
#         2) to mask the depth layers
#         3) to normalize the layers by using the low and high
#         4) to decide if the current layer needs to be returned
#     '''
#     for ii in range(start_idx, end_idx):
#         temp = depth.copy()
#
#         if ii > -1 and ii < K:
#             low = ii * layer_bin
#             if ii == K-1:
#                 high = np.max(depth)
#             else:
#                 high = (ii+1) * layer_bin
#
#             temp[temp < low] = low - 10    # if the target is on the boarder ?????
#             temp[temp > high] = high + 10  # acturaly no effect
#
#             if target_box is not None:
#                 target_values = target_patch.copy()
#                 target_values[target_values < low] = 0
#                 target_values[target_values > high] = 0
#                 target_values[target_values > 0] = 1
#                 target_in_layer[ii] = np.sum(target_values)
#
#             depth_layers[..., ii] = temp
#
#     return depth_layers, np.argmax(target_in_layer), K, layer_bin

def read_centered_layered_depth(file_name, target_box=None, depth_threshold=None, target_depth=None, radius=1000):
    '''
    return :
        - centered depth layer : target_depth - R : target_depth + R
        - the original depth
    '''
    # 1) get the depth image
    depth = cv2.imread(file_name, -1)
    depth = np.nan_to_num(depth, nan=0.0) # np.max(depth)

    H, W = depth.shape[0], depth.shape[1]
    depth_layers = np.zeros((H, W, 2), dtype=np.float32)

    # 2) decide the K and layer bin and the target depth
    if target_box is None and target_depth is None:
        print('Error : The target box or target_depth is required !!!')
        return None, None

    '''
    Find the possible target depth, by using the K-means,
        - assume that the init_box covers most the target, + few pixels belong to background(very large)
        - num_cluster = 2, target + background
        - calculate the most frequency pixels, which belong to target
        - if there is only the target in the box, then two centers will be similar, it is okay!
    '''
    if target_box is not None:
        target_box = [int(bb) for bb in target_box]
        target_depth = get_target_depth(depth, target_box)

    low = max(target_depth-radius, 0)
    high = target_depth + radius

    layer = depth.copy()
    layer[layer < low] = high + 10 #  low - 10    # if the target is on the boarder ?????
    layer[layer > high] = high + 10  # acturaly no effect

    depth_layers[..., 0] = layer

    # if target_depth > 10.0:
    #     max_depth = target_depth * 2.0 # ???
    #     depth[depth > max_depth] = max_depth
    depth_layers[..., 1] = depth

    return depth_layers, target_depth

def get_target_depth(depth, target_box):
    target_box = [int(bb) for bb in target_box]
    target_patch = depth[target_box[1]:target_box[1]+target_box[3], target_box[0]:target_box[0]+target_box[2]]
    # target_depth_values = target_patch.copy().reshape((-1, 1))

    # num_cluster = 2
    # kmeans = KMeans(init="random", n_clusters=num_cluster, n_init=10, max_iter=300, random_state=42)
    # kmeans.fit(target_depth_values)
    # depth_centers = kmeans.cluster_centers_
    # depth_labels = kmeans.labels_
    # depth_frequencies = []
    # for ii in range(num_cluster):
    #     depth_frequencies.append(np.sum(depth_labels[depth_labels==ii]))
    # target_depth = depth_centers[np.argmax(depth_frequencies)][0]


    target_depth_values = target_patch.flatten()
    # target_depth = np.mean(target_depth_values)

    target_depth_values.sort()
    if len(target_depth_values) > 0:
        target_depth = target_depth_values[int(len(target_depth_values)/2)]   # Song : middle value
        # target_depth = target_depth_values[int(len(target_depth_values)/3)]     # Song: assume that (foreground -> target depth -> backgorund)
        # target_depth = target_depth_values[int(len(target_depth_values)/3.5)]     # Song: assume that (foreground -> target depth -> backgorund)
        # target_depth = np.mean(target_depth_values[int(len(target_depth_values)/3.0) : int(len(target_depth_values)*2/3.0)])
    else:
        target_depth= 0.0

    return target_depth

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

def get_image(image_dir, dtype='rgb', normalize=True, depth_threshold=None,
              target_box=None, K=None, layer_bin=None, layer_index=None, target_depth=None, radius=1000):
    '''
    Parameters for the layered inputs , e.g. dtype= layered_colormap, layered_raw_depth, layered_normalized_depth
        -     K      : the number of layers to be splitted
        - layer_bin  : the distance of each layer
        - layer_index: the index of the layer containg the target
        - target_box : the gt box or the predicted box containing the possible target
    '''
    if dtype == 'rgb':
        image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
    elif dtype == 'raw_depth':
        image = read_depth(image_dir, normalize=False) # H * W * 3, [dp, dp, dp]
    elif dtype == 'normalized_depth':
        image = read_depth(image_dir, normalize=True) # H * W * 3, [dp, dp, dp] normalized
    elif dtype == 'colormap':
        image = read_colormap(image_dir, depth_threshold=depth_threshold) # H * W * 3
    elif dtype == 'raw_rgbd':
        image = read_rgbd(image_dir, normalize=False) # H * W * 4, rgb + raw_depth
    elif dtype == 'normalized_rgbd':
        image = read_rgbd(image_dir, normalize=True) # H * W * 4, rgb + normalized_depth
    elif dtype in ['centered_colormap', 'centered_raw_depth', 'centered_normalized_depth']:
        ''' 2) centered at previous prediction, e.g. Center-Radius <--> Center+Radius, + whole image'''
        image = read_centered_layered_depth(image_dir, depth_threshold=depth_threshold, target_depth=target_depth, target_box=target_box, radius=radius)         # H x W x K , k layers
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

def get_averageScore_from_history(scores, index, N=10):
    ''' get the average area of previous N frames '''
    avgScore = 0.0
    count = 0
    while count < N and index > -1:
        avgScore += scores[index]
        count += 1
        index -= 1

    return 1.0 * avgScore / count if count > 0 else 0.0

def get_layer_from_images(image, layer_index, dtype):

    temp_image = image[..., layer_index]

    if dtype == 'layered_colormap' or dtype == 'centered_colormap':
        temp_image = cv2.normalize(temp_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        temp_image = np.asarray(temp_image, dtype=np.uint8)
        temp_image = cv2.applyColorMap(temp_image, cv2.COLORMAP_JET)
    elif dtype == 'layered_raw_depth' or dtype == 'centered_raw_depth':
        temp_image = temp_image
        temp_image = cv2.merge((temp_image, temp_image, temp_image))
    elif dtype == 'layered_normalized_depth' or dtype == 'centered_normalized_depth':
        temp_image = cv2.normalize(temp_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        temp_image = np.asarray(temp_image, dtype=np.uint8)
        temp_image = cv2.merge((temp_image, temp_image, temp_image))
    else:
        print("Wrong dtype in get_layer_from_images")
        temp_image = None
    return temp_image

def search_layer(tracker, layer_index, image, p, prev_avgArea, prev_avgDepth, init_area, temp_results,
                 use_conf=True, use_max_min_scale=True, use_init_scale=True, use_depth_scale=False):

    # temp_region, temp_score_map, temp_iou, temp_score_max, temp_dis = temp_results
    ''' search in previous or next layer'''
    input_image = get_layer_from_images(image, layer_index, p.dtype)
    temp_results_in_layer = tracker.tracking(input_image, count=False)

    update = True
    ''' Consider the confidence '''
    if use_conf:
        conf_changes_in_layer = temp_results_in_layer[-2] > max(temp_results[-2], p.conf_threshold)
        update = update and  conf_changes_in_layer

    '''Consider the Area or Shape change'''
    temp_region_in_layer = temp_results_in_layer[0]
    temp_area_in_layer = temp_region_in_layer[2] * temp_region_in_layer[3]

    if use_max_min_scale:
        area_changes_in_layer = (1.0 * max(temp_area_in_layer, prev_avgArea) / (min(temp_area_in_layer, prev_avgArea)+1.0)) < p.area_scale_threshold
        update = update and area_changes_in_layer
    if use_init_scale:
        area_minimum_in_layer = temp_area_in_layer > init_area *  p.minimun_area_threshold
        update = update and area_minimum_in_layer

    ''' Consider the depth changes '''
    if use_depth_scale:
        target_depth_in_layer = get_target_depth(image[..., 1], temp_region_in_layer)
        print('target_depth_in_layer: ', target_depth_in_layer)
        target_depth_changes_in_layer = (1.0 * max(target_depth_in_layer, prev_avgDepth) / (min(target_depth_in_layer, prev_avgDepth)+1.0))  < p.target_depth_changes_threshold
        update = update and target_depth_changes_in_layer

    if update:
        temp_results = temp_results_in_layer
    print('- Move: ', update, ' layer_index: ', layer_index, ' score: ', temp_results_in_layer[-2], 'area : ', temp_area_in_layer, 'init_area: ', init_area)

    return temp_results, update

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
            image, init_target_depth = get_image(image_dir, dtype=p.dtype, depth_threshold=p.depth_threshold,
                                                 target_depth=None, target_box=region1, radius=p.radius)
        else:
            ''' return H*W*3 images, e.g. colormap, rgb, raw_depth, normalized depth , Or rgbd'''
            image = get_image(image_dir, dtype=p.dtype, depth_threshold=p.depth_threshold)

        h = image.shape[0]
        w = image.shape[1]

        box = np.array([region1[0] / w, region1[1] / h, (region1[0] + region1[2]) / w, (region1[1] + region1[3]) / h]) # w, h in (0 , 1)
        tic = time.time()

        if p.dtype in ['centered_colormap', 'centered_raw_depth', 'centered_normalized_depth']:
            input_image = get_layer_from_images(image, 0, p.dtype)
        else:
            input_image = image

        tracker = Dimp_LTMU_Tracker(input_image, region, p=p, groundtruth=groundtruth, tracker_name=tracker_name, tracker_params=tracker_params)

        ''' -------------- Song : we add some new parameters here ------------'''
        ''' Song : we copy a tracker here '''
        tracker_copy = tracker
        tracker_copy_available = True
        ''' Song : we keep the initial tracker for reset '''
        init_tracker = tracker
        reset = False
        ''' Song : we keep the init area of the target, if the current area < init_area*0.1 ? '''
        init_area = region1[2] * region1[3]
        init_depth = get_target_depth(image[..., 1], region1)
        ''' Song : we count the frames when the area is too small, e.g. occlusion, out-of-view, rotation'''
        num_small_area_frames = 0
        prev_small_area_idx = -1
        # ----------------------------------------------------------------------

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
                    while prev_score < p.conf_threshold and temp_im_id > 1:
                        temp_im_id -= 1
                        prev_depth = previous_depths[temp_im_id-1]
                        prev_score = previous_scores[temp_im_id-1]

                    n_history = 20
                    prev_avgScore = get_averageScore_from_history(previous_scores, im_id-1, N=n_history)
                    prev_avgArea = get_averageArea_from_history(previous_scores, bBoxes_results, im_id-1, N=n_history)
                    prev_avgDepth = get_averageDepth_from_history(previous_scores, previous_depths, im_id-1, N=n_history)
                else:
                    ''' Re-set the tracker , using the inital parameters '''
                    tracker = init_tracker
                    prev_depth = init_target_depth
                    prev_avgDepth = init_target_depth
                    prev_avgArea = init_area
                    prev_avgScore = 1.0
                    reset = False

                image, _, = get_image(image_dir, dtype=p.dtype, depth_threshold=p.depth_threshold,
                                      target_depth=prev_depth, target_box=None, radius=p.radius)
            else:
                image = get_image(image_dir, dtype=p.dtype, depth_threshold=p.depth_threshold)

            ''' Tracking process '''
            if p.dtype in ['centered_colormap', 'centered_raw_depth', 'centered_normalized_depth']:

                ''' 1) search the target in the optimal layer '''
                input_image = get_layer_from_images(image, 0, p.dtype)
                centered_temp_region, centered_score_map, centered_temp_iou, centered_temp_score_max, centered_temp_dis = tracker.tracking(input_image)
                temp_results = (centered_temp_region, centered_score_map, centered_temp_iou, centered_temp_score_max, centered_temp_dis)

                ''' Compare the predicted depth to the previous history depths ? or the previous depth ????'''
                temp_area = centered_temp_region[2] * centered_temp_region[3]
                temp_target_depth = get_target_depth(image[..., 1], centered_temp_region)

                '''
                How to decide that prediction is not reliable ???

                    - conf  > conf_threshold
                    - area  vs pre_area   or History area ??
                    - depth vs prev_depth or History depth ??
                '''
                temp_depth_changes_flag = 1.0 * max(temp_target_depth, prev_depth) / (min(temp_target_depth, prev_depth)+1.0) > p.target_depth_changes_threshold
                temp_area_changes_flag = temp_area < prev_avgArea * p.minimun_area_threshold
                temp_conf_changes_flag = centered_temp_score_max < p.conf_threshold

                if temp_conf_changes_flag or temp_area_changes_flag or temp_depth_changes_flag :

                    '''search the whole image'''
                    temp_results, update = search_layer(tracker, 1, image, p, prev_avgArea, prev_avgDepth, init_area, temp_results)

                    if update:
                        temp_target_depth = get_target_depth(image[..., 1], temp_results[0])

                    else:
                        if tracker_copy_available:
                            print('------------------------- using the tracker_copy --------------')
                            temp_results, update  = search_layer(tracker_copy, 0, image, p, prev_avgArea, prev_avgDepth, init_area, temp_results)

                            if update:
                                print('-----------------------------------------Rollback the tracker to previous temp --------------------------------------------------')
                                tracker = tracker_copy
                                tracker_copy_available = False
                                temp_target_depth = get_target_depth(image[..., 0], temp_results[0])

                            else:
                                temp_results, update  = search_layer(tracker_copy, 1, image, p, prev_avgArea, prev_avgDepth, init_area, temp_results)
                                if update:
                                    print('-----------------------------------------Rollback the tracker to previous temp --------------------------------------------------')
                                    tracker = tracker_copy
                                    tracker_copy_available = False
                                    temp_target_depth = get_target_depth(image[..., 1], temp_results[0])
                        #
                        # print('------------------------- using the init-trtacker --------------')
                        # temp_results, update  = search_layer(init_tracker, 0, image, p, prev_avgArea, prev_avgDepth, init_area, temp_results)
                        #
                        # if update:
                        #     print('-----------------------------------------Rollback the tracker to previous temp --------------------------------------------------')
                        #     tracker = init_tracker
                        #     # tracker_copy_available = False
                        #     temp_target_depth = get_target_depth(image[..., 0], temp_results[0])
                        #
                        # else:
                        #     temp_results, update  = search_layer(init_tracker, 1, image, p, prev_avgArea, prev_avgDepth, init_area, temp_results)
                        #     if update:
                        #         print('-----------------------------------------Rollback the tracker to previous temp --------------------------------------------------')
                        #         tracker = init_tracker
                        #         # tracker_copy_available = False
                        #         temp_target_depth = get_target_depth(image[..., 1], temp_results[0])
                else:

                    '''
                    How to decide that it is time to copy the tracker ??
                    '''
                    renew_tracker_copy = True
                    renew_tracker_copy = renew_tracker_copy and prev_avgScore > p.conf_rollback
                    renew_tracker_copy = renew_tracker_copy and temp_area > init_area * p.area_rollback    # area scales in a range
                    renew_tracker_copy = renew_tracker_copy and prev_avgArea > init_area * p.area_rollback  # compared to the inital area

                    if  renew_tracker_copy and im_id%p.rollback_iter==0:
                        print('------------------------Copy Tracker --------------------------')
                        tracker_copy = tracker
                        tracker_copy_available = True

                region, score_map, iou, score_max, dis = temp_results

                '''
                Here we should check the predicted_target_depth and the x-y location
                but how to use the x-y location

                '''

                temp_depth_changes_flag = 1.0 * max(temp_target_depth, prev_depth) / (min(temp_target_depth, prev_depth)+1.0)  > p.target_depth_changes_threshold

                if temp_depth_changes_flag:
                    print('Target moves too far... not be reliable , do not update, decrease the conf')
                    print('original : score_max = ', score_max)
                    score_max = 1.0 * min(temp_target_depth, prev_depth) / (max(temp_target_depth, prev_depth)+1.0)
                    print('new scores_max = ', score_max)
                    temp_target_depth = prev_depth


                ''' if the area change too much, discard..., update the score '''
                temp_area = region[2]*region[3]
                if temp_area < init_area * p.minimun_area_threshold:
                    print('!!!!!!!!!!!!! Frame %d : the predicted area is too small  %f vs init_area %f!!!!!!!!!!!!!!!'%(im_id, temp_area, init_area))

                    score_max = 1.0 * temp_area / init_area

                    if prev_small_area_idx == im_id - 1:
                        num_small_area_frames += 1
                        prev_small_area_idx = im_id

                    else:
                        prev_small_area_idx = im_id
                        num_small_area_frames = 1

                    if num_small_area_frames > 20:
                        print('!!!!!!!!!!!!!!! long term too small object, re-set the tracker to the inital one !!!!!!!!!!')
                        num_small_area_frames = 0
                        reset = True
                else:
                    num_small_area_frames = -1

                previous_depths[im_id] = temp_target_depth
                previous_scores[im_id] = score_max

                print("%d: " % seq_id + video + ": %d /" % im_id + "%d" % len(image_list)
                      + ' temp_target_depth : %f'%temp_target_depth + ' score: %f'%score_max
                      + ' area: %f'%(1.0 * region[2]*region[3]) + ' avg_previous_are : %s'%prev_avgArea
                      + ' init_area: %f'%init_area
                      + ' init_depth : %f'%init_target_depth)
            else:
                region, score_map, iou, score_max, dis = tracker.tracking(image)
                print("%d: " % seq_id + video + ": %d /" % im_id + "%d" % len(image_list) + "score: %f"%score_max)

            t = time.time() - tic
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
