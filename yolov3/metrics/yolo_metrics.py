import tensorflow as tf
from tensorflow.keras.metrics import binary_accuracy
from yolov3.losses import cal_iou

epsilon = 1e-07

def wrap_obj_acc(grid_shape, bbox_num, class_num):
    def obj_acc(y_true, y_pred):
        y_true = tf.reshape(
            y_true,
            (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
        y_pred = tf.reshape(
            y_pred,
            (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C
        
        c_true = y_true[..., 4] # N*S*S*1
        c_pred = tf.reduce_max(y_pred[..., 4], # N*S*S*B
                               axis=-1,
                               keepdims=True) # N*S*S*1

        bi_acc = binary_accuracy(c_true, c_pred)

        return bi_acc
    return obj_acc


def wrap_iou_acc(grid_shape, bbox_num, class_num):
    def iou_acc(y_true, y_pred):
        y_true = tf.reshape(
            y_true,
            (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
        y_pred = tf.reshape(
            y_pred,
            (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C

        pred_obj_mask = tf.cast(y_pred[..., 4] >= 0.5,
                                dtype=y_true.dtype) # N*S*S*B
        has_obj_mask = y_true[..., 4] # N*S*S*1
        has_obj_mask = has_obj_mask*pred_obj_mask # N*S*S*B

        xywh_true = y_true[..., :4] # N*S*S*1*4
        xywh_pred = y_pred[..., :4] # N*S*S*B*4

        iou_scores = cal_iou(xywh_true, xywh_pred, grid_shape) # N*S*S*B
        iou_scores = iou_scores*has_obj_mask # N*S*S*B

        total = tf.reduce_sum(pred_obj_mask)

        return tf.reduce_sum(iou_scores)/(total + epsilon)
    return iou_acc


def wrap_class_acc(grid_shape, bbox_num, class_num):
    def class_acc(y_true, y_pred):
        y_true = tf.reshape(
            y_true,
            (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
        y_pred = tf.reshape(
            y_pred,
            (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C
        
        pred_obj_mask = tf.cast(y_pred[..., 4] >= 0.5,
                                dtype=y_true.dtype) # N*S*S*B

        has_obj_mask = y_true[..., 4] # N*S*S*1
        has_obj_mask = has_obj_mask*pred_obj_mask # N*S*S*B

        pi_true = tf.argmax(y_true[..., -class_num:], # N*S*S*1*C
                            axis=-1) # N*S*S*1
        pi_pred = tf.argmax(y_pred[..., -class_num:], # N*S*S*B*C
                            axis=-1) # N*S*S*B
        
        equal_mask = tf.cast(pi_true == pi_pred,
                             dtype=y_true.dtype) # N*S*S*B
        equal_mask = equal_mask*has_obj_mask # N*S*S*B

        total = tf.reduce_sum(pred_obj_mask)

        return tf.reduce_sum(equal_mask)/(total + epsilon)
    return class_acc