import tensorflow as tf
from tensorflow.keras.metrics import binary_accuracy
from yolov1_5.losses import cal_iou

epsilon = 1e-07

def wrap_obj_acc(grid_shape, bbox_num, class_num):
    def obj_acc(y_true, y_pred):
        xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, *grid_shape, 1, 5)) # N*S*S*1*5
        xywhc_pred = tf.reshape(
            y_pred[..., :-class_num],
            (-1, *grid_shape, bbox_num, 5)) # N*S*S*B*5
        
        c_true = xywhc_true[..., 4] # N*S*S*1
        c_pred = tf.reduce_max(xywhc_pred[..., 4], # N*S*S*B
                               axis=-1,
                               keepdims=True) # N*S*S*1

        bi_acc = binary_accuracy(c_true, c_pred)

        return bi_acc
    return obj_acc


def wrap_mean_iou(grid_shape, bbox_num, class_num):
    def mean_iou(y_true, y_pred):
        xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, *grid_shape, 1, 5)) # N*S*S*1*5
        xywhc_pred = tf.reshape(
            y_pred[..., :-class_num],
            (-1, *grid_shape, bbox_num, 5)) # N*S*S*B*5

        has_obj_mask = xywhc_true[..., 4] # N*S*S*1
        
        iou_scores = cal_iou(xywhc_true, xywhc_pred, grid_shape) # N*S*S*B
        iou_scores = tf.reduce_max(iou_scores, axis=-1, keepdims=True) # N*S*S*1
        iou_scores = iou_scores*has_obj_mask # N*S*S*1

        num_p = tf.reduce_sum(has_obj_mask)

        return tf.reduce_sum(iou_scores)/(num_p + epsilon)
    return mean_iou


def wrap_class_acc(grid_shape, bbox_num, class_num):
    def class_acc(y_true, y_pred):
        xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, *grid_shape, 5)) # N*S*S*5

        has_obj_mask = xywhc_true[..., 4] # N*S*S

        pi_true = tf.argmax(y_true[..., -class_num:], # N*S*S*C
                            axis=-1) # N*S*S
        pi_pred = tf.argmax(y_pred[..., -class_num:], # N*S*S*C
                            axis=-1) # N*S*S

        equal_mask = tf.cast(pi_true == pi_pred,
                             dtype=y_true.dtype) # N*S*S
        equal_mask = equal_mask*has_obj_mask # N*S*S

        num_p = tf.reduce_sum(has_obj_mask)

        return tf.reduce_sum(equal_mask)/(num_p + epsilon)
    return class_acc


def wrap_recall(grid_shape, bbox_num, class_num, iou_threshold=0.5):
    def recall(y_true, y_pred):
        xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, *grid_shape, 1, 5)) # N*S*S*1*5
        xywhc_pred = tf.reshape(
            y_pred[..., :-class_num],
            (-1, *grid_shape, bbox_num, 5)) # N*S*S*B*5

        has_obj_mask = xywhc_true[..., 4] # N*S*S*1

        pi_true = tf.argmax(y_true[..., -class_num:], # N*S*S*C
                            axis=-1) # N*S*S
        pi_pred = tf.argmax(y_pred[..., -class_num:], # N*S*S*C
                            axis=-1) # N*S*S
        
        equal_mask = tf.cast(pi_true == pi_pred,
                             dtype=y_true.dtype) # N*S*S
        equal_mask = tf.expand_dims(equal_mask, axis=-1) # N*S*S*1
        equal_mask = equal_mask*has_obj_mask # N*S*S*1
        
        iou_scores = cal_iou(xywhc_true, xywhc_pred, grid_shape) # N*S*S*B
        iou_scores = iou_scores*equal_mask # N*S*S*B
        iou_scores = tf.reduce_max(iou_scores, axis=-1, keepdims=True) # N*S*S*1

        num_tp = tf.reduce_sum(
            tf.cast(iou_scores >= iou_threshold, dtype=y_true.dtype))
        num_p = tf.reduce_sum(has_obj_mask)

        return num_tp/(num_p + epsilon)
    return recall