import time
import torch
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
from pykeops.torch import LazyTensor

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64

def KMeans(x, K=10, Niter=10, verbose=False):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c

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

        # ori_colormap = depth.copy()
        # ori_colormap = cv2.normalize(ori_colormap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # ori_colormap = np.asarray(ori_colormap, dtype=np.uint8)
        # ori_colormap = cv2.applyColorMap(ori_colormap, cv2.COLORMAP_JET)
        #
        # stack_depth = depth.copy()
        # stack_depth = cv2.normalize(stack_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # stack_depth = np.asarray(stack_depth, dtype=np.uint8)
        # stack_depth = cv2.merge((stack_depth, stack_depth, stack_depth))
        #
        # groundtruth_box = groundtruth[frame_idx]
        # try:
        #     gt_box = [int(float(bb)) for bb in groundtruth_box.split(',')]
        # except:
        #     gt_box = [0, 0, 0, 0] # xywh

        '''
        How to decide the N and depth_bin ????
        if given the GT or the previous prediction - ???

        How to decide the N ?
            - for indoor, the farest distance < 10M
            - for outdoor, the farest distance > 10M
        '''
        # N = 5 # N layers
        # depth_bin = 1000 if np.max(depth) > 10000 else None # 10m outdoor
        # # print(N, depth_bin)
        # layers, layer_id = layered_depth_image(depth, N=N, depth_bin=depth_bin, box=gt_box)
        #
        # fig, axs = plt.subplots(2, 3)
        # fig.suptitle(sequence)
        # axs[0, 0].imshow(color)
        # axs[0, 1].imshow(ori_colormap)
        # axs[0, 2].imshow(stack_depth)
        #
        # for ii in range(0, 3):
        #     dp_layer = layers[..., ii+layer_id-1]
        #     dp_layer = cv2.normalize(dp_layer, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #     dp_layer = np.asarray(dp_layer, dtype=np.uint8)
        #     dp_layer = cv2.applyColorMap(dp_layer, cv2.COLORMAP_JET)
        #     axs[1, ii].imshow(dp_layer)
        #     axs[1, ii].title.set_text(str(ii+layer_id-1)+' -th layer')
        # plt.show()

        depth_values = depth.copy().reshape((-1,1))
        depth_values = np.asarray(depth_values, dtype=np.float32)
        depth_values = torch.from_numpy(depth_values)
        cl, c = KMeans(depth_values, K=10)
        print(cl, c)
