# Copyright 2020 Samson. All Rights Reserved.
# =============================================================================

"""Yolo V3.
"""

__version__ = "3.1"
__author__ = "Samson Woof"

from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..'))

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet152V2

from utils import tools
from .models import yolo_body, tiny_yolo_body
from .models import yolo_resnet90_body
from .models import yolo_keras_app_body
from .models import yolo_head
from .losses import wrap_yolo_loss
from .metrics import wrap_obj_acc, wrap_iou_acc, wrap_class_acc


class Acc_type(object):
    obj = "obj"
    iou = "iou"
    classes = "class"


class Yolo(object):
    """Yolo class.

    Use read_file() to read dataset、
    create_model() to create a tf.keras Model.
    Compile the model by using loss()、metrics().
    Use vis_img to visualize the images and annotations. 

    Args:
        input_shape: A tuple of 3 integers,
            shape of input image.
        class_names: A list, containing all label names.
        anchors: 2D array like,
            the anchors will be divided up evenly
            across scales.

    Attributes:
        input_shape
        grid_num: An integer,
            the input image will be divided into 
            the square of this grid number.
        abox_num: An integer, the number of anchor boxes.
        class_names
        class_num: An integer, the number of all classes.
        model: A tf.keras Model.
    """

    def __init__(self,
                 input_shape=(416, 416, 3),
                 class_names=[],
                 anchors=[[0.89663461, 0.78365384],
                          [0.37500000, 0.47596153],
                          [0.27884615, 0.21634615],
                          [0.14182692, 0.28605769],
                          [0.14903846, 0.10817307],
                          [0.07211538, 0.14663461],
                          [0.07932692, 0.05528846],
                          [0.03846153, 0.07211538],
                          [0.02403846, 0.03125000]]):
        self.input_shape = input_shape
        self.grid_num = input_shape[0]//32
        self.abox_num = len(anchors)//3
        self.class_names = class_names
        self.class_num = len(class_names)
        self.anchors = anchors
        self.model = None
        
    def create_model(self, backbone="full_darknet",
                     pretrained_weights=None,
                     pretrained_body="imagenet"):
        """Create a yolo model.

        Args:
            backbone: A string,
                one of "full_darknet"、"tiny_darknet"、
                "resnet90"、"resnet50v2"、"resnet101v2"、
                "resnet152v2".
            pretrained_weights: A string, 
                file path of pretrained model.
            pretrained_body: A tf.keras Model,
                allow any input shape model.

        Returns:
            A tf.keras Model.
        """
        if isinstance(pretrained_body, str):
            pre_body_weights = pretrained_body
            pretrained_body = None
        else:
            pre_body_weights = None
        
        if backbone == "full_darknet":
            model_body = yolo_body(self.input_shape,
                pretrained_weights=pre_body_weights)
        elif backbone == "tiny_darknet":
            model_body = tiny_yolo_body(self.input_shape)
        elif backbone == "resnet90":
            model_body = yolo_resnet90_body(self.input_shape)
        elif backbone == "resnet50v2":
            model_body = yolo_keras_app_body(ResNet50V2,
                self.input_shape, fpn_id=[143, 75], 
                pretrained_weights=pre_body_weights)
        elif backbone == "resnet101v2":
            model_body = yolo_keras_app_body(ResNet101V2,
                self.input_shape, fpn_id=[143, 75], 
                pretrained_weights=pre_body_weights)
        elif backbone == "resnet152v2":
            model_body = yolo_keras_app_body(ResNet152V2,
                self.input_shape, fpn_id=[143, 75], 
                pretrained_weights=pre_body_weights)
        else:
            raise ValueError("Invalid backbone: %s" % backbone)
        if pretrained_body is not None:
            model_body.set_weights(pretrained_body.get_weights())
        self.model = yolo_head(model_body,
                               self.class_num,
                               self.anchors)  
        if pretrained_weights is not None:
            self.model.load_weights(pretrained_weights)
        self.grid_num = self.model.output[0].shape[1]
        self.abox_num = len(self.anchors)//len(self.model.output)

    def read_file(self, img_path=None, label_path=None,
                  label_format="labelimg",
                  augmenter=None,
                  aug_times=1,
                  shuffle=True, seed=None,
                  thread_num=10,
                  fpn_id=0):
        """Read the images and annotaions created by labelimg or labelme.

        Args:
            img_path: A string, 
                file path of images.
            label_path: A string,
                file path of annotations.
            label_format: A string,
                one of "labelimg" and "labelme".
            augmenter: A `imgaug.augmenters.meta.Sequential` instance.
            aug_times: An integer,
                the default is 1, which means no augmentation.
            shuffle: Boolean, default: True.
            seed: An integer, random seed, default: None.
            thread_num: An integer,
                specifying the number of threads to read files.
            fpn_id: An integer,
                specifying the layer index of FPN.
                The id of smallest feature layer is 0.

        Returns:
            A tuple of 2 ndarrays, (data, label),
            shape of data: (batch_size, img_height, img_width, channel)
            shape of label: (batch_size, grid_num, grid_num, info)
        """
        return tools.read_file(img_path=img_path, 
                               label_path=label_path,
                               label_format=label_format,
                               size=self.input_shape[:2], 
                               grid_num=self.grid_num*(2**fpn_id),
                               class_names=self.class_names,
                               augmenter=augmenter,
                               aug_times=aug_times,
                               shuffle=shuffle,
                               seed=seed,
                               thread_num=thread_num)

    def vis_img(self, img, *label_datas,
                conf_threshold=0.5,
                nms_mode=0,
                nms_threshold=0.5,
                nms_sigma=0.5,
                **kwargs):
        """Visualize the images and annotaions by pyplot.

        Args:
            img: A ndarray of shape(img_height, img_width, channel).
            *label_datas: Ndarrays of shape(grid_num, grid_num, info).
                Multiple label data can be given at once.
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
                             *label_datas, 
                             class_names=self.class_names,
                             conf_threshold=conf_threshold,
                             nms_mode=nms_mode,  
                             nms_threshold=nms_threshold,
                             nms_sigma=nms_sigma,                     
                             version=3,
                             **kwargs)

    def loss(self,
             binary_weight=1,
             loss_weight=[1, 1, 1, 1],
             ignore_thresh=.6,
             use_focal_loss=False,
             focal_loss_gamma=2,
             use_scale=True,
             fpn_id=0):
        """Loss of yolo.

        Args:
            binary_weight: A list,
                the ratio of positive and negative.
            loss_weight: A dictionary or list of length 4,
                specifying the weights of each loss.
                Example:{"xy":1, "wh":1, "conf":1, "pr":1},
                or [5, 5, 0.5, 1].
            ignore_thresh: A float,
                threshold of ignoring the false positive.
            use_focal_loss: A boolean,
                whether use focal loss or not, default is False.
            focal_loss_gamma: A integer.
            use_scale: A boolean,
                whether use scale factor or not, default is True.
            fpn_id: An integer,
                specifying the layer index of FPN.
                The id of smallest feature layer is 0.

        Returns:
            A loss function conforming to tf.keras specification.
        """
        if isinstance(loss_weight, dict):
            loss_weight_list = []
            loss_weight_list.append(loss_weight["xy"])
            loss_weight_list.append(loss_weight["wh"])
            loss_weight_list.append(loss_weight["conf"])
            loss_weight_list.append(loss_weight["pr"])
            loss_weight = loss_weight_list
        anchors_id = self.abox_num*fpn_id
        return wrap_yolo_loss(
            grid_num=self.grid_num*(2**fpn_id),
            bbox_num=self.abox_num, 
            class_num=self.class_num,
            anchors=self.anchors[
                anchors_id:anchors_id + self.abox_num],
            binary_weight=binary_weight,
            loss_weight=loss_weight,
            ignore_thresh=ignore_thresh,
            use_focal_loss=use_focal_loss,
            focal_loss_gamma=focal_loss_gamma,
            use_scale=use_scale)
    
    def metrics(self, acc_type="obj", fpn_id=0):
        """Metrics of yolo.

        Args:
            acc_type: A string,
                one of "obj" or "iou" or "class".
            fpn_id: An integer,
                specifying the layer index of FPN.
                The id of smallest feature layer is 0.

        Returns:
            A metric function conforming to tf.keras specification.
        """        
        if acc_type == "obj":
            return wrap_obj_acc(self.grid_num*(2**fpn_id),
                                self.abox_num, 
                                self.class_num)
        elif acc_type == "iou":
            return wrap_iou_acc(self.grid_num*(2**fpn_id),
                                self.abox_num, 
                                self.class_num)
        elif acc_type == "class":
            return wrap_class_acc(self.grid_num*(2**fpn_id),
                                  self.abox_num,
                                  self.class_num)