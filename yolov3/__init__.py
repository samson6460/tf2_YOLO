# Copyright 2020 Samson. All Rights Reserved.
# =============================================================================

"""Yolo V3.
"""

__version__ = "3.1"
__author__ = "Samson Woof"

from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..'))

from collections.abc import Iterable
from tensorflow.keras.utils import Sequence
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


class Yolov3DataSequence(Sequence):
    def __init__(self, seq, fpn_layers):
        self.seq = seq
        self.fpn_layers = fpn_layers
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, idx):
        img_data, label_data = self.seq[idx]
        label_list = [label_data]
        for _ in range(self.fpn_layers - 1):
            label_data = tools.down2xlabel(label_data)
            label_list.insert(0, label_data)
        return img_data, label_list


class Yolo(object):
    """Yolo class.

    Use read_file() to read dataset、
    create_model() to create a tf.keras Model.
    Compile the model by using loss()、metrics().
    Use vis_img() to visualize the images and annotations. 

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
        file_names: A list of string
            with all file names that have been read.
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
        self.fpn_layers = 3
        self.anchors = anchors
        self.model = None
        self.file_names = None
        
    def create_model(self, backbone="full_darknet",
                     pretrained_weights=None,
                     pretrained_body="pascal_voc"):
        """Create a yolo model.

        Args:
            backbone: A string,
                one of "full_darknet"、"tiny_darknet"、
                "resnet90"、"resnet50v2"、"resnet101v2"、
                "resnet152v2".
            pretrained_weights: A string, 
                file path of pretrained model.
            pretrained_body: None、"pascal_voc"(only for `full_darknet`)
                、"imagenet"(only for `resnetXXX`) or tf.keras model
                which allows any input shape.

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
        self.fpn_layers = len(self.model.output)
        self.abox_num = len(self.anchors)//self.fpn_layers

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
            A tuple: (img, label_list)(ndarray, list),
            shape of img: (batch_size, img_height, img_width, channel)
            shape of label: (batch_size, grid_num, grid_num, info)
            label_list contains the label of all FPN layers.
        """
        img_data, label_data, path_list = tools.read_file(
            img_path=img_path, 
            label_path=label_path,
            label_format=label_format,
            size=self.input_shape[:2], 
            grid_num=self.grid_num*(2**(self.fpn_layers - 1)),
            class_names=self.class_names,
            rescale=rescale,
            preprocessing=preprocessing,
            augmenter=augmenter,
            aug_times=aug_times,
            shuffle=shuffle, seed=seed,
            encoding=encoding,
            thread_num=thread_num)
        self.file_names = path_list

        label_list = [label_data]
        for _ in range(self.fpn_layers - 1):
            label_data = tools.down2xlabel(label_data)
            label_list.insert(0, label_data)

        return img_data, label_list

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
                Sequence[i]: (img, label_list)(ndarray, list),
            shape of img: (batch_size, img_height, img_width, channel)
            shape of label: (batch_size, grid_num, grid_num, info)
            label_list contains the label of all FPN layers.
        """
        seq = tools.YoloDataSequence(
            img_path=img_path,
            label_path=label_path,
            batch_size=batch_size,
            label_format=label_format,
            size=self.input_shape[:2],
            rescale=rescale,
            preprocessing=preprocessing,
            grid_num=self.grid_num*(2**(self.fpn_layers - 1)),
            class_names=self.class_names,
            augmenter=augmenter,
            shuffle=shuffle,
            seed=seed,
            encoding=encoding,
            thread_num=thread_num)
        self.file_names = seq.path_list

        v3seq = Yolov3DataSequence(seq, self.fpn_layers)

        return v3seq

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
        return tools.vis_img(
            img, 
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
             use_scale=True):
        """Loss of yolo.

        Args:
            binary_weight: A float or a list of float,
                the ratio of positive and negative for each FPN layers.
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

        Returns:
            A loss function conforming to tf.keras specification.
        """
        if (not isinstance(binary_weight, Iterable)
            or len(binary_weight) != self.fpn_layers):
            binary_weight = [binary_weight]*self.fpn_layers
        
        if isinstance(loss_weight, dict):
            loss_weight_list = []
            loss_weight_list.append(loss_weight["xy"])
            loss_weight_list.append(loss_weight["wh"])
            loss_weight_list.append(loss_weight["conf"])
            loss_weight_list.append(loss_weight["pr"])
            loss_weight = loss_weight_list
        
        loss_list = []
        for fpn_id in range(self.fpn_layers):
            anchors_id = self.abox_num*fpn_id
            loss_list.append(wrap_yolo_loss(
                grid_num=self.grid_num*(2**fpn_id),
                bbox_num=self.abox_num, 
                class_num=self.class_num,
                anchors=self.anchors[
                    anchors_id:anchors_id + self.abox_num],
                binary_weight=binary_weight[fpn_id],
                loss_weight=loss_weight,
                ignore_thresh=ignore_thresh,
                use_focal_loss=use_focal_loss,
                focal_loss_gamma=focal_loss_gamma,
                use_scale=use_scale))
        return loss_list
    
    def metrics(self, acc_type="obj"):
        """Metrics of yolo.

        Args:
            acc_type: A string,
                one of "obj"、"iou" or "class".
                Or "obj+iou"、"obj+iou+class" to specify
                multiple metrics.

        Returns:
            A metric function conforming to tf.keras specification.
        """
        metrics_list = [[] for _ in range(self.fpn_layers)]
        if "obj" in acc_type:
            for fpn_id in range(self.fpn_layers):
                metrics_list[fpn_id].append(
                    wrap_obj_acc(
                        self.grid_num*(2**fpn_id),
                        self.abox_num, 
                        self.class_num))
        if "iou" in acc_type:
            for fpn_id in range(self.fpn_layers):
                metrics_list[fpn_id].append(
                    wrap_iou_acc(
                        self.grid_num*(2**fpn_id),
                        self.abox_num, 
                        self.class_num))
        if "class" in acc_type:
            for fpn_id in range(self.fpn_layers):
                metrics_list[fpn_id].append(
                    wrap_class_acc(
                        self.grid_num*(2**fpn_id),
                        self.abox_num, 
                        self.class_num))
        return metrics_list