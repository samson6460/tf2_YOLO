# Copyright 2020 Samson. All Rights Reserved.
# =============================================================================

"""Yolo V2.
"""

__version__ = "3.1"
__author__ = "Samson Woof"

from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..'))

from tensorflow.keras.models import load_model

from utils import tools
from .models import yolo_body, yolo_head
from .losses import wrap_yolo_loss
from .metrics import wrap_obj_acc, wrap_iou_acc, wrap_class_acc


class Acc_type(object):
    obj = "obj"
    iou = "iou"
    classes = "class"


class Yolo(object):
    """Yolo class.

    Use read_file_to_dataset() to read dataset、
    Use vis_img() to visualize the images and annotations. 
    create_model() to create a tf.keras Model.
    Compile the model by using loss()、metrics(). 

    Args:
        input_shape: A tuple of 3 integers,
            shape of input image.
        class_names: A list, containing all label names.
        anchors: 2D array like.
    
    Attributes:
        input_shape
        class_names
        grid_num: An integer,
            the input image will be divided into 
            the square of this grid number.
        abox_num: An integer, the number of anchor boxes.
        class_num: An integer, the number of all classes.
        anchors: 2D array like, prior anchor boxes.
        model: A tf.keras Model.
        file_names: A list of string
            with all file names that have been read.
    """

    def __init__(self,
                 input_shape=(416, 416, 3),
                 class_names=[]):
        self.input_shape = input_shape
        self.grid_num = input_shape[0]//32
        self.abox_num = 5
        self.class_names = class_names
        self.class_num = len(class_names)
        self.anchors = None
        self.model = None
        self.file_names = None
        
    def create_model(self,
                     anchors=[[0.75157846, 0.70525231],
                              [0.60637077, 0.27136769],
                              [0.25680231, 0.42110308],
                              [0.14418923, 0.15865615],
                              [0.04405615, 0.05210654]],
                     backbone="darknet",
                     pretrained_weights=None,
                     pretrained_backbone=None):
        """Create a yolo model.

        Args:
            anchors: 2D array like, 
                prior anchor boxes(widths, heights),
                all the values should be normalize to 0-1.
            backbone:: A string,
                one of "darknet"、"unet"、"mobilenet".
            pretrained_weights: A string,
                file path of pretrained model.
            pretrained_backbone: "imagenet"(only for `mobilenet`)
                or a tf.keras model which allows any input shape.

        Returns:
            A tf.keras Model.
        """
        model_body = yolo_body(self.input_shape,
                               backbone,
                               pretrained_backbone)

        self.model = yolo_head(model_body,
                               self.class_num,
                               anchors)  
        if pretrained_weights is not None:
            self.model.load_weights(pretrained_weights)
        self.anchors = anchors
        self.abox_num = len(anchors)
        self.grid_num = self.model.output.shape[1]

    def read_file_to_dataset(
        self, img_path=None, label_path=None,
        label_format="labelimg",
        rescale=1/255,
        preprocessing=None,
        augmenter=None,
        aug_times=1,
        shuffle=True, seed=None,
        encoding="big5",
        thread_num=10):
        """Read the images and annotaions
        created by labelimg or labelme as ndarray.

        Args:
            img_path: A string, 
                file path of images.
            label_path: A string,
                file path of annotations.
            label_format: A string,
                one of "labelimg" and "labelme".
            rescale: A float or None,
                specifying how the image value should be scaled.
                If None, no scaled.
            preprocessing: A function of data preprocessing,
                (e.g. noralization, shape manipulation, etc.)
            augmenter: A `imgaug.augmenters.meta.Sequential` instance.
            aug_times: An integer,
                the default is 1, which means no augmentation.
            shuffle: Boolean, default: True.
            seed: An integer, random seed, default: None.
            encoding: A string,
                encoding format of file,
                default: "big5".
            thread_num: An integer,
                specifying the number of threads to read files.

        Returns:
            A tuple of 2 ndarrays, (img, label),
            - shape of img: (batch_size, img_height, img_width, channel)
            - shape of label: (batch_size, grid_num, grid_num, info)
        """
        img_data, label_data, path_list = tools.read_file(
            img_path=img_path, 
            label_path=label_path,
            label_format=label_format,
            size=self.input_shape[:2], 
            grid_num=self.grid_num,
            class_names=self.class_names,
            rescale=rescale,
            preprocessing=preprocessing,
            augmenter=augmenter,
            aug_times=aug_times,
            shuffle=shuffle, seed=seed,
            encoding=encoding,
            thread_num=thread_num)
        self.file_names = path_list

        return img_data, label_data

    def read_file_to_sequence(
        self, img_path=None,
        label_path=None,
        batch_size=20,
        label_format="labelimg",
        rescale=1/255,
        preprocessing=None,
        augmenter=None,
        shuffle=True, seed=None,
        encoding="big5",
        thread_num=1):
        """Read the images and annotaions
        created by labelimg or labelme as sequence.

        Args:
            img_path: A string, 
                file path of images.
            label_path: A string,
                file path of annotations.
            batch_size: An integer,
                size of the batches of data (default: 20).
            label_format: A string,
                one of "labelimg" and "labelme".
            rescale: A float or None,
                specifying how the image value should be scaled.
                If None, no scaled.
            preprocessing: A function of data preprocessing,
                (e.g. noralization, shape manipulation, etc.)
            augmenter: A `imgaug.augmenters.meta.Sequential` instance.
            shuffle: Boolean, default: True.
            seed: An integer, random seed, default: None.
            encoding: A string,
                encoding format of file,
                default: "big5".
            thread_num: An integer,
                specifying the number of threads to read files.

        Returns:
            A tf.Sequence: 
                Sequence[i]: (img, label)
            - shape of img: (batch_size, img_height, img_width, channel)
            - shape of label: (batch_size, grid_num, grid_num, info)
        """
        seq = tools.YoloDataSequence(
            img_path=img_path,
            label_path=label_path,
            batch_size=batch_size,
            label_format=label_format,
            size=self.input_shape[:2],
            rescale=rescale,
            preprocessing=preprocessing,
            grid_num=self.grid_num,
            class_names=self.class_names,
            augmenter=augmenter,
            shuffle=shuffle,
            seed=seed,
            encoding=encoding,
            thread_num=thread_num)
        self.file_names = seq.path_list

        return seq

    def vis_img(self, img, label_data,
                conf_threshold=0.5,
                nms_mode=0,
                nms_threshold=0.5,
                nms_sigma=0.5,
                **kwargs):
        """Visualize the images and annotaions by pyplot.

        Args:
            img: A ndarray of shape(img_height, img_width, channel).
            label_data: A ndarray of shape(grid_num, grid_num, info).
            conf_threshold: A float,
                threshold for quantizing output.
            nms_mode: An integer,
                0: Not use NMS.
                1: Use NMS.
                2: Use Soft-NMS.
            nms_threshold: A float,
                threshold for eliminating duplicate boxes.
            nms_sigma: A float,
                sigma for Soft-NMS.
                threshold for eliminating duplicate boxes.
            figsize: (float, float), optional, default: None
                width, height in inches. If not provided, defaults to [6.4, 4.8].
            axis: bool or str
                If a bool, turns axis lines and labels on or off.
                If a string, possible values are:
                https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.axis.html
            savefig_path: None or string or PathLike or file-like object
                A path, or a Python file-like object.
            connection: A string,
                one of "head"、"tail",
                connect visualization of ground truth and prediction.
            fig_ax: (matplotlib.pyplot.figure, matplotlib.pyplot.axes),
                This argument only works
                when the connection is specified as "tail".            
            point_radius: 5
            point_color: "r"
            box_linewidth: 2
            box_color: "auto"
            text_color: "w"
            text_padcolor: "auto"
            text_fontsize: 12
        """
        return tools.vis_img(img, 
                             label_data, 
                             class_names=self.class_names,
                             conf_threshold=conf_threshold,
                             nms_mode=nms_mode,  
                             nms_threshold=nms_threshold,
                             nms_sigma=nms_sigma,                             
                             version=2,
                             **kwargs)

    def loss(self, binary_weight=1,
             loss_weight=[1, 1, 5, 1],
             ignore_thresh=0.6):
        """Loss of yolo.

        Args:
            binary_weight: A list,
                the ratio of positive and negative.
            loss_weight: A dictionary or list of length 4,
                specifying the weights of each loss.
                Example:{"xy":1, "wh":1, "conf":1, "prob":1},
                or [5, 5, 0.5, 1].
            ignore_thresh: A float,
                threshold of ignoring the false positive.

        Returns:
            A loss function conforming to tf.keras specification.
        """
        if isinstance(loss_weight, dict):
            loss_weight_list = []
            loss_weight_list.append(loss_weight["xy"])
            loss_weight_list.append(loss_weight["wh"])
            loss_weight_list.append(loss_weight["conf"])
            loss_weight_list.append(loss_weight["prob"])
            loss_weight = loss_weight_list
        return wrap_yolo_loss(
            grid_num=self.grid_num,
            bbox_num=self.abox_num, 
            class_num=self.class_num,
            anchors=self.anchors,
            binary_weight=binary_weight,
            loss_weight=loss_weight,
            ignore_thresh=ignore_thresh)
    
    def metrics(self, acc_type="obj"):
        """Metrics of yolo.

        Args:
            acc_type: A string,
                one of "obj" or "iou" or "class".

        Returns:
            A metric function conforming to tf.keras specification.
        """        
        if acc_type == "obj":
            return wrap_obj_acc(self.grid_num, 
                                self.abox_num, 
                                self.class_num)
        elif acc_type == "iou":
            return wrap_iou_acc(self.grid_num, 
                                self.abox_num, 
                                self.class_num)
        elif acc_type == "class":
            return wrap_class_acc(self.grid_num, 
                                  self.abox_num,
                                  self.class_num)