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

    # qg_rcnn_r50_fpn_path = '/home/yan/Data2/LTMU-rgbd-models/qg_rcnn_r50_fpn_2x_20181010-443129e1.pth'

class VOTLT_Results_Saver(object):
    def __init__(self, save_path, video, t):
        result_path = os.path.join(save_path, 'longterm')
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

def read_layeredColormap(file_name, target_box=None, depth_threshold=None, N=3):
    depth = cv2.imread(file_name, -1)
    if depth_threshold is not None:
        try:
            max_depth = np.max(depth)
            # print('max_depth : ', max_depth)
            if max_depth > depth_threshold:
                depth[depth > depth_threshold] = depth_threshold
        except:
            depth = depth

    '''
     To split the depth values :
        - 1) fixed  bins, e.g. 0-1000, 1000-2000, 2000-3000, 3000-4000, ....
        - 2) k-clustered based bins, e.g. by performing the K-clusters, to get the depth seeds, each seed is the center of layers
                                     e.g. K-Cluster(depth) -> dp = 30, 1000, 4500, 8000 -> Layer_30, Layer_1000, .... Layer_dp

    We use the 1) now !!!!!!, one problem, K and Step affect the performance, especially for indoor scene, needs the optimal K and Step
    '''
    H, W = depth.shape
    K = 10
    Step = 1000 # 1 meter
    depth_layers = np.zeros((H, W, K*3), dtype=np.float32)
    for ii in range(K):
        temp = depth.copy()

        '''
            1) to decide the boarders, low and high
            2) to mask the depth layers
            3) to normalize the layers by using the low and high
            4) to decide if the current layer needs to be returned
        '''
        low = ii * Step
        if ii == K-1:
            high = np.max(depth)
        else:
            high = (ii+1) * Step
        temp[temp < low] = low - 10    # if the target is on the boarder ?????
        temp[temp > high] = high + 10  # acturaly no effect

        temp = cv2.normalize(temp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        temp = np.asarray(temp, dtype=np.uint8)
        temp = cv2.applyColorMap(temp, cv2.COLORMAP_JET)

        depth_layers[..., ii*3:(ii+1)*3] = temp

    return depth_layers

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
    return rgbd # np.asarray(rgbd, dtype=np.float32) # Song : remember to check the value in depth change or not , from uint8 to 32f ???

def get_image(image_dir, dtype='rgb', normalize=True, depth_threshold=None, target_box=None):

    if dtype == 'rgb':
        image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
    elif dtype == 'depth':
        image = read_depth(image_dir, normalize=normalize) # H * W * 3, [dp, dp, dp]
    elif dtype == 'colormap':
        image = read_colormap(image_dir, depth_threshold=depth_threshold) # H * W * 3
    elif dtype == 'rgbd':
        image = read_rgbd(image_dir, normalize=normalize) # H * W * 4
    elif dtype == 'layered_colormap':
        ''' return H x W x (3*C) , three layers '''
        image = read_layeredColormap(image_dir, target_box=target_box, depth_threshold=depth_threshold)
    elif dtype == 'layered_depth':
        '''return H x W x (3*C), three layers '''
        image = read_layeredDepth(image_dir, depth_threshold=depth_threshold)
    else:
        print('unknown input type : %s'%dtype)
        image = None

    return image

def run_seq_list(Dataset, p, sequence_list, data_dir, tracker_name='dimp', tracker_params='dimp50', dtype='rgb', depth_threshold=None):
    '''
    Song :
        - dtype : the data type for inputs, e.g. rgb, colormap, depth, rgbd
            - rgb      : H x W x 3
            - depth    : 1) normalzie to [0, 255], 2) concatenate the channels , [depth, depth, depth]
            - colormap : 1) normalzie depth images, 2) convert to colormap, JET
            - rgbd     : H x W x (3+1), rgb+depth, [..., :3] = rgb, [..., 3:] = depth

            - layeredColormap :
            - layeredDepth :
    '''

    m_shape = 19
    base_save_path = os.path.join('./results', p.name, Dataset)
    if not os.path.exists(base_save_path):
        if p.save_results and not os.path.exists(os.path.join(base_save_path, 'eval_results')):
            os.makedirs(os.path.join(base_save_path, 'eval_results'))
        if p.save_training_data and not os.path.exists(os.path.join(base_save_path, 'train_data')):
            os.makedirs(os.path.join(base_save_path, 'train_data'))

    for seq_id, video in enumerate(sequence_list):
        sequence_dir, groundtruth = get_groundtruth(Dataset, data_dir, video)

        if p.save_training_data:
            result_save_path = os.path.join(base_save_path, 'train_data', video + '.txt')
            if os.path.exists(result_save_path):
                continue
        if p.save_results:
            result_save_path = os.path.join(base_save_path, 'eval_results', video + '.txt')
            if os.path.exists(result_save_path):
                continue

        if dtype == 'rgbd':
            color_image_list = os.listdir(sequence_dir['color'])
            color_image_list.sort()
            image_list = [im[:-4] for im in color_image_list if im.endswith("jpg") or im.endswith("jpeg") or im.endswith("png")]
            image_dir = {}
            image_dir['color'] = sequence_dir['color'] + image_list[0] + '.jpg'
            image_dir['depth'] = sequence_dir['depth'] + image_list[0] + '.png'
        else:
            image_list = os.listdir(sequence_dir)
            image_list.sort()
            image_list = [im for im in image_list if im.endswith("jpg") or im.endswith("jpeg") or im.endswith("png")]
            image_dir = sequence_dir + image_list[0]

        image = get_image(image_dir, dtype=dtype, depth_threshold=depth_threshold)
        h = image.shape[0]
        w = image.shape[1]

        region = Region(groundtruth[0, 0], groundtruth[0, 1], groundtruth[0, 2], groundtruth[0, 3])
        region1 = groundtruth[0]

        box = np.array([region1[0] / w, region1[1] / h, (region1[0] + region1[2]) / w, (region1[1] + region1[3]) / h]) # w, h in (0 , 1)
        tic = time.time()
        tracker = Dimp_LTMU_Tracker(image, region, p=p, groundtruth=groundtruth, tracker_name=tracker_name, tracker_params=tracker_params)
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

        for im_id in range(1, len(image_list)):
            tic = time.time()

            if dtype == 'rgbd':
                image_dir = {}
                image_dir['color'] = sequence_dir['color'] + image_list[im_id] + '.jpg'
                image_dir['depth'] = sequence_dir['depth'] + image_list[im_id] + '.png'
            else:
                image_dir = sequence_dir + image_list[im_id]
                # print(im_id, image_dir)

            image = get_image(image_dir, dtype=dtype, depth_threshold=depth_threshold)
            # image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
            print("%d: " % seq_id + video + ": %d /" % im_id + "%d" % len(image_list))
            region, score_map, iou, score_max, dis = tracker.tracking(image)
            print('score: ', score_max)
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
        if p.save_results:
            np.savetxt(os.path.join(base_save_path, 'eval_results', video + '.txt'), bBoxes_results,
                       fmt="%.8f,%.8f,%.8f,%.8f")

        tracker.sess.close()
        tf.reset_default_graph()


def eval_tracking(Dataset, p, mode=None, video=None, tracker_name='dimp', tracker_params='dimp50', dtype=None):

    if not dtype:
        # dtype in [depth, colormap, rgbd, rgb, layered_depth, layered_colormap]
        dtype = Dataset[5:] if Dataset in ['cdtb_depth', 'cdtb_colormap','cdtb_rgbd'] else 'rgb'

    depth_threshold = 10000 if Dataset in ['cdtb_depth', 'cdtb_colormap', 'cdtb_rgbd'] else None
    # depth_threshold = None

    if Dataset == 'lasot':
        classes = os.listdir(lasot_dir)
        classes.sort()
        for c in classes:
            sequence_list, data_dir = get_seq_list(Dataset, mode=mode, classes=c)
            run_seq_list(Dataset, p, sequence_list, data_dir, tracker_name=tracker_name, tracker_params=tracker_params, dtype=dtype, depth_threshold=depth_threshold)
    elif Dataset in ['votlt18', 'votlt19', 'tlp', 'otb', 'demo', 'cdtb_depth', 'cdtb_colormap', 'cdtb_color', 'cdtb_rgbd']:
        sequence_list, data_dir = get_seq_list(Dataset, video=video)
        run_seq_list(Dataset, p, sequence_list, data_dir, tracker_name=tracker_name, tracker_params=tracker_params, dtype=dtype, depth_threshold=depth_threshold)
    else:
        print('Warning: Unknown dataset.')
