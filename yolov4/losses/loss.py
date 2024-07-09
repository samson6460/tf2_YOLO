"""YOLOv4 loss."""

import math
import tensorflow as tf
import numpy as np

EPSILON = 1e-07


def cal_iou(xywh_true, xywh_pred, grid_shape, return_ciou=False):
    """Calculate IOU of two tensors.
    return shape: (N, S, S, B)[, (N, S, S, B)]
    """
    grid_shape = np.array(grid_shape[::-1])
    xy_true = xywh_true[..., 0:2]/grid_shape # N*S*S*1*2
    wh_true = xywh_true[..., 2:4]

    xy_pred = xywh_pred[..., 0:2]/grid_shape # N*S*S*B*2
    wh_pred = xywh_pred[..., 2:4]

    half_wh_true = wh_true / 2.
    mins_true    = xy_true - half_wh_true
    maxes_true   = xy_true + half_wh_true

    half_wh_pred = wh_pred / 2.
    mins_pred    = xy_pred - half_wh_pred
    maxes_pred   = xy_pred + half_wh_pred

    intersect_mins  = tf.maximum(mins_pred,  mins_true)
    intersect_maxes = tf.minimum(maxes_pred, maxes_true)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = wh_true[..., 0] * wh_true[..., 1]
    pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = intersect_areas/(union_areas + EPSILON)

    if return_ciou:
        enclose_mins = tf.minimum(mins_pred,  mins_true)
        enclose_maxes = tf.maximum(maxes_pred, maxes_true)

        enclose_wh = enclose_maxes - enclose_mins
        enclose_c2 = (tf.pow(enclose_wh[..., 0], 2)
                      + tf.pow(enclose_wh[..., 1], 2))

        p_rho2 = (tf.pow(xy_true[..., 0] - xy_pred[..., 0], 2)
                  + tf.pow(xy_true[..., 1] - xy_pred[..., 1], 2))

        atan_true = tf.atan(wh_true[..., 0] / (wh_true[..., 1] + EPSILON))
        atan_pred = tf.atan(wh_pred[..., 0] / (wh_pred[..., 1] + EPSILON))

        v_nu = 4.0 / (math.pi ** 2) * tf.pow(atan_true - atan_pred, 2)
        a_alpha = v_nu / (1 - iou_scores + v_nu)

        ciou_scores = iou_scores - p_rho2/enclose_c2 - a_alpha*v_nu

        return iou_scores, ciou_scores

    return iou_scores


def wrap_yolo_loss(grid_shape,
                   bbox_num,
                   class_num,
                   anchors=None,
                   binary_weight=1,
                   loss_weight=[1, 1, 1],
                   wh_reg_weight=0.01,
                   ignore_thresh=.6,
                   truth_thresh=1,
                   label_smooth=0,
                   focal_loss_gamma=2):
    """Wrapped YOLOv4 loss function."""
    def yolo_loss(y_true, y_pred):
        if anchors is None:
            panchors = 1
        else:
            panchors = tf.reshape(anchors, (1, 1, 1, bbox_num, 2))

        y_true = tf.reshape(
            y_true,
            (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*(5+C)
        y_pred = tf.reshape(
            y_pred,
            (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*(5+C)

        xywh_true = y_true[..., :4] # N*S*S*1*4
        xywh_pred = y_pred[..., :4] # N*S*S*B*4

        iou_scores, ciou_scores = cal_iou(
            xywh_true, xywh_pred, grid_shape, return_ciou=True) # N*S*S*B

        response_mask = tf.one_hot(tf.argmax(iou_scores, axis=-1),
                                   depth=bbox_num,
                                   dtype=xywh_true.dtype) # N*S*S*B

        has_obj_mask = y_true[..., 4]*response_mask # N*S*S*B

        if truth_thresh < 1:
            truth_mask = tf.cast(
                iou_scores > truth_thresh,
                iou_scores.dtype) # N*S*S*B
            has_obj_mask = has_obj_mask + truth_mask*(1 - has_obj_mask)
        has_obj_mask_exp = tf.expand_dims(has_obj_mask, axis=-1) # N*S*S*B*1

        no_obj_mask = tf.cast(
            iou_scores < ignore_thresh,
            iou_scores.dtype) # N*S*S*B
        no_obj_mask = (1 - has_obj_mask)*no_obj_mask # N*S*S*B

        box_loss = tf.reduce_sum(
            tf.reduce_mean(
            has_obj_mask # N*S*S*B
            *(1 - ciou_scores), # N*S*S*B
            axis=0))

        c_pred = y_pred[..., 4] # N*S*S*B
        c_pred = tf.clip_by_value(c_pred, EPSILON, 1 - EPSILON)

        if label_smooth > 0:
            obj_error = tf.math.abs(1 - label_smooth - c_pred)
            no_obj_error = tf.math.abs(label_smooth - c_pred)
        else:
            obj_error = 1 - c_pred
            no_obj_error = c_pred

        has_obj_c_loss = -tf.reduce_sum(
            tf.reduce_mean(
            has_obj_mask # N*S*S*B
            *(obj_error**focal_loss_gamma)
            *tf.math.log(1 - obj_error),
            axis=0))

        no_obj_c_loss = -tf.reduce_sum(
            tf.reduce_mean(
            no_obj_mask # N*S*S*B
            *(no_obj_error**focal_loss_gamma)
            *tf.math.log(1 - no_obj_error),
            axis=0))

        c_loss = has_obj_c_loss + binary_weight*no_obj_c_loss

        p_true = y_true[..., -class_num:] # N*S*S*1*C
        p_pred = y_pred[..., -class_num:] # N*S*S*B*C

        p_pred = tf.clip_by_value(p_pred, EPSILON, 1 - EPSILON)
        p_loss = -tf.reduce_sum(
            tf.reduce_mean(
            has_obj_mask_exp # N*S*S*B*1
            *(p_true*tf.math.log(p_pred)
              + (1 - p_true)*tf.math.log(1 - p_pred)), # N*S*S*B*C
            axis=0))

        wh_pred = y_pred[..., 2:4]/panchors # N*S*S*B*2
        wh_pred = tf.math.log(wh_pred) # N*S*S*B*2

        wh_reg = tf.reduce_sum(
            tf.reduce_mean(wh_pred**2, axis=0))

        loss = (loss_weight[0]*box_loss
                + loss_weight[1]*c_loss
                + loss_weight[2]*p_loss
                + wh_reg_weight*wh_reg)

        return loss

    return yolo_loss
