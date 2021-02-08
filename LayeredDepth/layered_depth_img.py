import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

def  layered_depth_image(dp_img, N = 10, depth_bin=1000, box = None):
    np.nan_to_num(dp_img, nan=0)

    h, w = dp_img.shape
    if N is None and depth_bin is not None :
        max_depth = np.max(dp_img)
        N = max_depth // depth_slots + 1
    if N is not None and depth_bin is None:
        max_depth = np.max(dp_img)
        depth_bin = max_depth / N
    if N is None and depth_bin is None:
        print('at least provide one of N and depth_bin')
        return

    if box is not None:
        gt_patch = dp_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

        '''
        Find the possible target depth
        '''
        gt_depth_values = gt_patch.copy().reshape((-1, 1))
        kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(gt_depth_values)
        depth_centers = kmeans.cluster_centers_
        depth_labels = kmeans.labels_
        depth_frequencies = []
        for ii in range(3):
            depth_frequencies.append(np.sum(depth_labels[depth_labels==ii]))
        target_depth = depth_centers[np.argmax(depth_frequencies)][0]

    layers = np.zeros((h, w, N), dtype=np.float32)
    target_in_layer = np.zeros((N,), dtype=np.float32)

    '''
    K-cluster, to estimate the depth seeds, then to re-order the layers , each layer contains a depth center ?
    '''
    # depth_values = dp_img.copy().reshape((-1, 1))
    # kmeans = KMeans(init="random", n_clusters=N, n_init=10, max_iter=300, random_state=42)
    # kmeans.fit(depth_values)
    # depth_centers = kmeans.cluster_centers_
    # depth_labels = kmeans.labels_
    # print(depth_centers)

    for ii in range(N):
        temp = dp_img.copy()
        low = ii * depth_bin
        if ii < N-1:
            high = (ii+1) * depth_bin
        else:
            high = np.max(dp_img)
        temp[temp < low] = low
        temp[temp > high] = high

        if box is not None:
            gt_values = gt_patch.copy()
            gt_values[gt_values < low] = 0
            gt_values[gt_values > high] = 0
            gt_values[gt_values > 0] = 1
            target_in_layer[ii] = np.sum(gt_values)

        layers[..., ii] = temp

    return layers, np.argmax(target_in_layer)

if __name__ == '__main__':

    sequence = 'cartman'
    sequence_path = '/home/yan/Data2/vot-workspace/sequences/%s/'%sequence
    groundtruth_path = sequence_path + 'groundtruth.txt'
    with open(groundtruth_path, 'r') as fp:
        groundtruth = fp.readlines()
    groundtruth = [line.strip() for line in groundtruth]

    num_frames = len(os.listdir(sequence_path+'color/'))
    for frame_idx in range(1, 2) :# num_frames+1, 20):

        color_frame = sequence_path + 'color/%08d.jpg'%frame_idx
        depth_frame = sequence_path + 'depth/%08d.png'%frame_idx

        color = cv2.cvtColor(cv2.imread(color_frame), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_frame, -1)

        ori_colormap = depth.copy()
        ori_colormap = cv2.normalize(ori_colormap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        ori_colormap = np.asarray(ori_colormap, dtype=np.uint8)
        ori_colormap = cv2.applyColorMap(ori_colormap, cv2.COLORMAP_JET)

        stack_depth = depth.copy()
        stack_depth = cv2.normalize(stack_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        stack_depth = np.asarray(stack_depth, dtype=np.uint8)
        stack_depth = cv2.merge((stack_depth, stack_depth, stack_depth))

        groundtruth_box = groundtruth[frame_idx]
        try:
            gt_box = [int(float(bb)) for bb in groundtruth_box.split(',')]
        except:
            gt_box = [0, 0, 0, 0] # xywh

        '''

        How to decide the N and depth_bin ????
        if given the GT or the previous prediction - ???

        How to decide the N ?
            - for indoor, the farest distance < 10M
            - for outdoor, the farest distance > 10M

        '''
        N = 5 # N layers
        depth_bin = 1000 if np.max(depth) > 10000 else None # 10m outdoor
        # print(N, depth_bin)
        layers, layer_id = layered_depth_image(depth, N=N, depth_bin=depth_bin, box=gt_box)

        fig, axs = plt.subplots(2, 3)
        fig.suptitle(sequence)
        axs[0, 0].imshow(color)
        axs[0, 1].imshow(ori_colormap)
        axs[0, 2].imshow(stack_depth)

        for ii in range(0, 3):
            dp_layer = layers[..., ii+layer_id-1]
            dp_layer = cv2.normalize(dp_layer, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            dp_layer = np.asarray(dp_layer, dtype=np.uint8)
            dp_layer = cv2.applyColorMap(dp_layer, cv2.COLORMAP_JET)
            axs[1, ii].imshow(dp_layer)
            axs[1, ii].title.set_text(str(ii+layer_id-1)+' -th layer')
        plt.show()
