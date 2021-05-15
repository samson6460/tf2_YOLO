import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt


def iou(center_boxes, data_boxes):
    """Calculate IOU of boxes.
    """
    center_w = center_boxes[..., 0]
    center_h = center_boxes[..., 1]
    center_area = center_w*center_h

    data_w = data_boxes[..., 0]
    data_h = data_boxes[..., 1]
    data_area = data_w*data_h

    intersect_area = np.minimum(center_area, data_area)
    union_area = np.maximum(center_area, data_area)
    iou_scores = intersect_area/union_area

    return iou_scores


def iou_dist(center_boxes, data_boxes):
    """Calculate IOU distance.

    IOU distance = 1 - IOU.
    """
    dist = 1 - iou(center_boxes, data_boxes)
    return dist


def euclidean_dist(center_boxes, data_boxes):
    """Calculate euclidean distance.
    """
    dist = np.sqrt(np.sum(np.square(center_boxes - data_boxes), axis=-1))
    return dist


def kmeans(data, n_cluster, dist_func,
           stop_dist, max_iternum=10000,
           verbose=True):
    """Calculate k-means clustering.
    
    Args:
        data: A 2D ndarray like, 
            shape: (num_samples, num_dims).
        n_cluster: An integer,
            specifing how many groups
            to divide into.
        dist_func: A function,
            specifing the distance function.
        stop_dist: A float,
            the distance to stop iteration.
        max_iternum: An integer,
            the maximum number of iterations.
        verbose: An boolean,
            whether to display iterative information,
            default is True.

    Returns:
        A 2D ndarray,
            shape: (n_cluster, num_dims).
    """
    n_dim = data.shape[-1]
    data = np.expand_dims(data, axis=0)
    data_max = data.max()
    data_min = data.min()

    center = rand(n_cluster*n_dim).reshape(
        (n_cluster, 1, n_dim))*data.max()
    center = center*(data_max - data_min) + data_min

    epoch = 1
    while True:
        dist = dist_func(center, data)
        dist_argmin = np.argmin(dist, axis=0)
        new_center = np.copy(center)

        for n in range(n_cluster):
            index = np.where(dist_argmin == n)[0]
            if len(index) > 0:
                clusters = data[0, index]
                cluster = np.mean(clusters, axis=0)
            else:
                cluster = rand(n_dim)*(data_max - data_min) + data_min
            new_center[n, 0] = cluster

        loss = np.mean(dist_func(center, new_center))
        center = new_center
        if verbose:
            print("epoch %2d: loss = %.4f" % (epoch, loss))
        epoch += 1
        if loss < stop_dist or epoch > max_iternum:
            break

    center = center.reshape((n_cluster, n_dim))
    center = center.astype("float32")
    return center

if __name__ == '__main__':
    n_data = 100
    data_boxes = rand(n_data*2).reshape((n_data, 2))

    center = kmeans(data_boxes,
                    n_cluster=3,
                    dist_func=euclidean_dist,
                    stop_dist=0.01)

    plt.scatter(data_boxes[..., 0], data_boxes[..., 1])
    plt.scatter(center[..., 0],
                center[..., 1],
                c="black")
    plt.show()

    center = kmeans(data_boxes,
                    n_cluster=3,
                    dist_func=iou_dist,
                    stop_dist=0.01)

    plt.scatter(data_boxes[..., 0], data_boxes[..., 1])
    plt.scatter(center[..., 0],
                center[..., 1],
                c="black")
    plt.show()