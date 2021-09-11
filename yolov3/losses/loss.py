import tensorflow as tf
import numpy as np

epsilon = 1e-07


def cal_iou(xywh_true, xywh_pred, grid_shape):
    grid_shape = np.array(grid_shape[::-1])
    xy_true = xywh_true[..., 0:2]/grid_shape # N*S*S*1*2
    wh_true = xywh_true[..., 2:4]

    xy_pred = xywh_pred[..., 0:2]/grid_shape # N*S*S*B*2
    wh_pred = xywh_pred[..., 2:4]
    
    half_xy_true = wh_true / 2.
    mins_true    = xy_true - half_xy_true
    maxes_true   = xy_true + half_xy_true

    half_xy_pred = wh_pred / 2.
    mins_pred    = xy_pred - half_xy_pred
    maxes_pred   = xy_pred + half_xy_pred       
    
    intersect_mins  = tf.maximum(mins_pred,  mins_true)
    intersect_maxes = tf.minimum(maxes_pred, maxes_true)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = wh_true[..., 0] * wh_true[..., 1]
    pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = (intersect_areas + epsilon)/(union_areas + epsilon)
    
    return iou_scores


def wrap_yolo_loss(grid_shape,
                   bbox_num,
                   class_num,
                   anchors=None,
                   binary_weight=1,
                   loss_weight=[1, 1, 1, 1],
                   ignore_thresh=.6,
                   use_focal_loss=False,
                   focal_loss_gamma=2,
                   use_scale=True):
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

        iou_scores = cal_iou(xywh_true, xywh_pred, grid_shape) # N*S*S*B

        response_mask = tf.one_hot(tf.argmax(iou_scores, axis=-1),
                                   depth=bbox_num,
                                   dtype=xywh_true.dtype) # N*S*S*B

        has_obj_mask = y_true[..., 4]*response_mask # N*S*S*B
        has_obj_mask_exp = tf.expand_dims(has_obj_mask, axis=-1) # N*S*S*B*1

        no_obj_mask = tf.cast(
            iou_scores < ignore_thresh,
            iou_scores.dtype) # N*S*S*B
        no_obj_mask = (1 - has_obj_mask)*no_obj_mask # N*S*S*B

        xy_true = y_true[..., 0:2] # N*S*S*1*2
        xy_pred = y_pred[..., 0:2] # N*S*S*B*2

        wh_true = tf.maximum(y_true[..., 2:4]/panchors, epsilon) # N*S*S*1*2
        wh_pred = y_pred[..., 2:4]/panchors
        
        wh_true = tf.math.log(wh_true) # N*S*S*B*2
        wh_pred = tf.math.log(wh_pred) # N*S*S*B*2

        c_pred = y_pred[..., 4] # N*S*S*B

        if use_scale:
            box_loss_scale = 2 - y_true[..., 2:3]*y_true[..., 3:4] # N*S*S*1*1
        else:
            box_loss_scale = 1

        xy_loss = tf.reduce_sum(
            tf.reduce_mean(
            has_obj_mask_exp # N*S*S*B*1
            *box_loss_scale # N*S*S*1*1
            *tf.square(xy_true - xy_pred), # N*S*S*B*2
            axis=0))

        wh_loss = tf.reduce_sum(
            tf.reduce_mean(
            has_obj_mask_exp # N*S*S*B*1
            *box_loss_scale # N*S*S*1*1
            *tf.square(wh_true - wh_pred), # N*S*S*B*2
            axis=0))

        if use_focal_loss:
            c_pred = tf.clip_by_value(c_pred, epsilon, 1 - epsilon)

            has_obj_c_loss = -tf.reduce_sum(
                tf.reduce_mean(
                has_obj_mask # N*S*S*B
                *((1 - c_pred)**focal_loss_gamma)
                *tf.math.log(c_pred),
                axis=0))
            
            no_obj_c_loss = -tf.reduce_sum(
                tf.reduce_mean(
                no_obj_mask # N*S*S*B
                *((c_pred)**focal_loss_gamma)
                *tf.math.log(1 - c_pred),
                axis=0))
        else:
            has_obj_c_loss = tf.reduce_sum(
                tf.reduce_mean(
                has_obj_mask # N*S*S*1
                *(tf.square(1 - c_pred)), # N*S*S*B
                axis=0))

            no_obj_c_loss = tf.reduce_sum(
                tf.reduce_mean(
                no_obj_mask # N*S*S*1
                *(tf.square(0 - c_pred)), # N*S*S*B
                axis=0))
        
        c_loss = has_obj_c_loss + binary_weight*no_obj_c_loss

        p_true = y_true[..., -class_num:] # N*S*S*1*C
        p_pred = y_pred[..., -class_num:] # N*S*S*B*C

        p_pred = tf.clip_by_value(p_pred, epsilon, 1 - epsilon)
        p_loss = -tf.reduce_sum(
            tf.reduce_mean(
            has_obj_mask_exp # N*S*S*B*1
            *(p_true*tf.math.log(p_pred)
              + (1 - p_true)*tf.math.log(1 - p_pred)), # N*S*S*B*C
            axis=0))

        regularizer = tf.reduce_sum(
            tf.reduce_mean(wh_pred**2, axis=0))*0.01

        loss = (loss_weight[0]*xy_loss
                + loss_weight[1]*wh_loss
                + loss_weight[2]*c_loss
                + loss_weight[3]*p_loss
                + regularizer)

        return loss

    return yolo_loss