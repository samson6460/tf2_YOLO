import tensorflow as tf
from tensorflow.keras.metrics import binary_accuracy
from yolov1_5.losses import cal_iou

epsilon = 1e-07

def wrap_obj_acc(grid_num, bbox_num, class_num):
    def obj_acc(y_true, y_pred):
        xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, grid_num, grid_num, 1, 5)) # N*S*S*1*5
        xywhc_pred = tf.reshape(
            y_pred[..., :-class_num],
            (-1, grid_num, grid_num, bbox_num, 5)) # N*S*S*B*5
        
        c_true = xywhc_true[..., 4] # N*S*S*1
        c_pred = tf.reduce_max(xywhc_pred[..., 4], # N*S*S*B
                               axis=-1,
                               keepdims=True) # N*S*S*1

        bi_acc = binary_accuracy(c_true, c_pred)

        return bi_acc
    return obj_acc


def wrap_iou_acc(grid_num, bbox_num, class_num):
    def iou_acc(y_true, y_pred):
        xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, grid_num, grid_num, 1, 5)) # N*S*S*1*5
        xywhc_pred = tf.reshape(
            y_pred[..., :-class_num],
            (-1, grid_num, grid_num, bbox_num, 5)) # N*S*S*B*5

        pred_obj_mask = tf.cast(xywhc_pred[..., 4] >= 0.5,
                                dtype=y_true.dtype) # N*S*S*B
        has_obj_mask = xywhc_true[..., 4] # N*S*S*1
        has_obj_mask = has_obj_mask*pred_obj_mask
        
        iou_scores = cal_iou(xywhc_true, xywhc_pred, grid_num) # N*S*S*B
        iou_scores = iou_scores*has_obj_mask # N*S*S*B

        total = tf.reduce_sum(has_obj_mask)

        return tf.reduce_sum(iou_scores)/(total + epsilon)
    return iou_acc


def wrap_class_acc(grid_num, bbox_num, class_num):
    def class_acc(y_true, y_pred):
        xywhc_pred = tf.reshape(
            y_pred[..., :-class_num],
            (-1, grid_num, grid_num, bbox_num, 5)) # N*S*S*B*5

        pred_obj_mask = tf.reduce_max(xywhc_pred[..., 4], # N*S*S*B
                                      axis=-1) # N*S*S
        pred_obj_mask = tf.cast(pred_obj_mask >= 0.5,
                                dtype=y_true.dtype) # N*S*S

        pi_true = tf.argmax(y_true[..., -class_num:], # N*S*S*C
                            axis=-1) # N*S*S
        pi_pred = tf.argmax(y_pred[..., -class_num:], # N*S*S*C
                            axis=-1) # N*S*S

        equal_mask = tf.cast(pi_true == pi_pred,
                             dtype=y_true.dtype) # N*S*S
        equal_mask = equal_mask*pred_obj_mask # N*S*S

        total = tf.reduce_sum(pred_obj_mask)

        return tf.reduce_sum(equal_mask)/total
    return class_acc