import tensorflow as tf
import numpy as np

epsilon = 1e-07


def cal_iou(xywh_true, xywh_pred, grid_shape):
    grid_shape = np.array(grid_shape[::-1])
    xy_true = xywh_true[..., 0:2]/grid_shape # N*S*S*1*2
    wh_true = xywh_true[..., 2:4] # N*S*S*1*2

    xy_pred = xywh_pred[..., 0:2]/grid_shape # N*S*S*B*2
    wh_pred = xywh_pred[..., 2:4] # N*S*S*B*2
    
    half_xy_true = wh_true / 2. # N*S*S*1*2
    mins_true    = xy_true - half_xy_true
    maxes_true   = xy_true + half_xy_true

    half_xy_pred = wh_pred / 2. # N*S*S*B*2
    mins_pred    = xy_pred - half_xy_pred
    maxes_pred   = xy_pred + half_xy_pred       
    
    intersect_mins  = tf.maximum(mins_pred,  mins_true) # N*S*S*B*2
    intersect_maxes = tf.minimum(maxes_pred, maxes_true)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1] # N*S*S*B
    
    true_areas = wh_true[..., 0] * wh_true[..., 1] # N*S*S*1
    pred_areas = wh_pred[..., 0] * wh_pred[..., 1] # N*S*S*B

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = intersect_areas/(union_areas + epsilon)
    
    return iou_scores


def wrap_yolo_loss(grid_shape,
                   bbox_num,
                   class_num,
                   binary_weight=1,
                   loss_weight=[1, 1, 1, 1]):
    def yolo_loss(y_true, y_pred):   
        xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, *grid_shape, 1, 5)) # N*S*S*1*5
        xywhc_pred = tf.reshape(
            y_pred[..., :-class_num],
            (-1, *grid_shape, bbox_num, 5)) # N*S*S*B*5

        iou_scores = cal_iou(xywhc_true, xywhc_pred, grid_shape) # N*S*S*B
        response_mask = tf.one_hot(tf.argmax(iou_scores, axis=-1),
                                   depth=bbox_num,
                                   dtype=xywhc_true.dtype) # N*S*S*B
        response_mask_exp = tf.expand_dims(response_mask, axis=-1) # N*S*S*B*1

        has_obj_mask = xywhc_true[..., 4] # N*S*S*1
        has_obj_mask_exp = tf.expand_dims(has_obj_mask, axis=-1) # N*S*S*1*1
        no_obj_mask = 1 - has_obj_mask*response_mask # N*S*S*B

        xy_true = xywhc_true[..., 0:2] # N*S*S*1*2
        xy_pred = xywhc_pred[..., 0:2] # N*S*S*B*2

        wh_true = tf.maximum(xywhc_true[..., 2:4], epsilon) # N*S*S*1*2
        wh_pred = tf.maximum(xywhc_pred[..., 2:4], epsilon) # N*S*S*B*2

        c_pred = xywhc_pred[..., 4] # N*S*S*B
        
        xy_loss = tf.reduce_sum(
            tf.reduce_mean(
                has_obj_mask_exp # N*S*S*1*1
                *response_mask_exp # N*S*S*B*1
                *tf.square(xy_true - xy_pred), # N*S*S*B*2
                axis=0))

        wh_loss = tf.reduce_sum(
            tf.reduce_mean(
                has_obj_mask_exp # N*S*S*1*1
                *response_mask_exp # N*S*S*B*1
                *tf.square(tf.sqrt(wh_true) - tf.sqrt(wh_pred)), # N*S*S*B*2
                axis=0))

        has_obj_c_loss = tf.reduce_sum(
            tf.reduce_mean(
                has_obj_mask # N*S*S*1
                *response_mask # N*S*S*B
                *tf.square(iou_scores - c_pred), # N*S*S*B
                axis=0))

        no_obj_c_loss = tf.reduce_sum(
            tf.reduce_mean(
                no_obj_mask # N*S*S*1
                *tf.square(0 - c_pred), # N*S*S*B
                axis=0))
        
        c_loss = has_obj_c_loss + binary_weight*no_obj_c_loss

        p_true = y_true[..., -class_num:] # N*S*S*C
        p_pred = y_pred[..., -class_num:] # N*S*S*C
        p_pred = tf.clip_by_value(p_pred, epsilon, 1 - epsilon)

        p_loss = -tf.reduce_sum(
            tf.reduce_mean(
                has_obj_mask # N*S*S*1
                *p_true*tf.math.log(p_pred), # N*S*S*C
                axis=0))

        loss = (loss_weight[0]*xy_loss
                + loss_weight[1]*wh_loss
                + loss_weight[2]*c_loss
                + loss_weight[3]*p_loss)

        return loss

    return yolo_loss