# Copyright 2023 Samson. All Rights Reserved.
# =============================================================================

"""YOLO V4.
"""

__version__ = "1.0"
__author__ = "Samson Woof"

from collections.abc import Iterable
from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..'))

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet152V2

from utils import tools
from .models import yolo_body
from .models import yolo_keras_app_body
from .models import yolo_head
from .losses import wrap_yolo_loss
from .metrics import wrap_obj_acc, wrap_mean_iou
from .metrics import wrap_class_acc, wrap_recall


class MetricKind(object):
    """names of metric kind"""
    obj_acc = "obj_acc"
    mean_iou = "mean_iou"
    class_acc = "class_acc"
    recall = "recall"


class _Yolov4DataSequence(Sequence):
    def __init__(self, seq, pan_layers):
        self.seq = seq
        self.pan_layers = pan_layers
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, idx):
        img_data, label_data = self.seq[idx]
        label_list = [label_data]
        for _ in range(self.pan_layers - 1):
            label_data = tools.down2xlabel(label_data)
            label_list.insert(0, label_data)
        return img_data, label_list


class Yolo(object):
    """Yolo class.

    1. Use read_file_to_dataset() or read_file_to_sequence to read dataset.
    2. Use vis_img() to visualize the images and annotations. 
    3. Use create_model() to create a tf.keras Model.
    4. Compile the model by using loss(), metrics().

    Args:
        input_shape: A tuple of 3 integers,
            shape of input image.
        class_names: A list, containing all label names.
        anchors: 2D array like,
            the anchors will be divided up evenly
            across scales.

    Attributes:
        input_shape
        class_names
        grid_shape: A tuple or list of integers(height, width), 
            input images will be divided into 
            grid_shape[0] x grid_shape[1] grids.
        abox_num: An integer, the number of anchor boxes.
        class_num: An integer, the number of all classes.
        pan_layers: An integer, the number of fpn layers.
        anchors: 2D array like, prior anchor boxes.
        model: A tf.keras Model.
        file_names: A list of string
            with all file names that have been read.
    """

    def __init__(self,
                 input_shape=(608, 608, 3),
                 class_names:list=[]):
        self.input_shape = input_shape
        self.grid_shape = input_shape[0]//32, input_shape[1]//32
        self.abox_num = 3
        self.class_names = class_names
        self.class_num = len(class_names)
        self.pan_layers = 3
        self._model = None
        self._file_names = None
        self._anchors_trainable = False

    @property
    def model(self):
        """Yolo.model"""
        if self._model is None:
            raise ValueError(
               "You haven't created a model by using create_model().")
        return self._model

    @model.setter
    def model(self, _):
        raise ValueError(
            "Can't set attribute directly, "
            "please create a model by using create_model().")

    @model.deleter
    def model(self):
        del self._model
        self._model = None

    @property
    def anchors(self):
        """Yolo.anchors"""
        if self._model is None:
            raise ValueError(
               "To get anchors, you have to create a model first.")

        _anchors = []
        for i_out in range(self.pan_layers):
            for i_box in range(self.abox_num):
                l_name = f"out{i_out + 1}_box{i_box + 1}_anchor"
                _anchors.append(self.model.get_layer(
                    name=l_name).get_weights()[0])
        _anchors = np.squeeze(np.vstack(_anchors)).tolist()
        return _anchors

    @anchors.setter
    def anchors(self, anchor_boxes):
        for i_out in range(self.pan_layers):
            start_i = i_out*self.abox_num
            for i_box, box in enumerate(
                    anchor_boxes[start_i:start_i + self.abox_num]):
                self.model.get_layer(
                    name=f"out{i_out + 1}_box{i_box + 1}_anchor").set_weights(
                    [np.expand_dims(box, axis=((0, 1, 2)))]
                )

    @property
    def anchors_trainable(self):
        """Yolo.anchors_trainable"""
        return self._anchors_trainable

    @anchors_trainable.setter
    def anchors_trainable(self, trainable):
        for i_out in range(self.pan_layers):
            for i_box in range(self.abox_num):
                layer = self.model.get_layer(
                    name=f"out{i_out + 1}_box{i_box + 1}_anchor")
                layer.trainable = trainable

        self._anchors_trainable = trainable

    @property
    def file_names(self):
        """Yolo.file_names"""
        if self._file_names is None:
            raise ValueError(
               "You haven't read files.")
        return self._file_names

    def reshape_anchors(self, ori_shape, shape=None):
        """Reshape the model anchors.

        Args:
            ori_shape: The original shape(width, height).
            shape: The shape to convert(width, height).
                If the argument is ignored,
                input shape of model will be used.
        """
        if shape is None:
            shape = self.input_shape[1::-1]
        grid_amp = ori_shape[0]/shape[0], ori_shape[1]/shape[1]

        for i_out in range(self.pan_layers):
            for i_box in range(self.abox_num):
                layer = self.model.get_layer(
                    name=f"out{i_out + 1}_box{i_box + 1}_anchor")
                layer.set_weights(
                    [layer.get_weights()[0]*grid_amp]
                )

    def create_model(self,
                     anchors=None,
                     backbone="csp_darknet",
                     pretrained_weights=None,
                     pretrained_body="ms_coco"):
        """Create a yolo model.

        Args:
            anchors: 2D array like, 
                prior anchor boxes(width, height),
                all the values should be normalize to 0-1.
                If using pretrained weights, this argument can be ignored.
            backbone: A string,
                one of "csp_darknet", 
                "resnet50", "resnet101", "resnet152",
                "resnet50v2", "resnet101v2", "resnet152v2".
            pretrained_weights: A string, 
                file path of pretrained model.
            pretrained_body: None、"ms_coco"(only for `csp_darknet`),
                "imagenet"(only for `resnetXXX`) or tf.keras model
                which allows input shape of (32x, 32x, 3).

        Returns:
            A tf.keras Model.
        """
        use_arg_anchors = True
        if pretrained_weights is None:
            if anchors is None:
                raise ValueError(
                    "Without pretrained weights, `anchors` can't be empty.")
        else:
            pretrained_body = None
            if anchors is None:
                anchors = [[1, 1] for _ in range(
                    self.pan_layers*self.abox_num)]
                use_arg_anchors = False

        if isinstance(pretrained_body, str):
            str_body_weights = pretrained_body
            pretrained_body = None
        else:
            str_body_weights = None

        if backbone == "csp_darknet":
            model_body = yolo_body(self.input_shape,
                pretrained_weights=str_body_weights)
        elif backbone == "resnet50":
            model_body = yolo_keras_app_body(ResNet50,
                self.input_shape, pan_ids=[-33, 80],
                pretrained_weights=str_body_weights)
        elif backbone == "resnet101":
            model_body = yolo_keras_app_body(ResNet101,
                self.input_shape, pan_ids=[-33, 80],
                pretrained_weights=str_body_weights)
        elif backbone == "resnet152":
            model_body = yolo_keras_app_body(ResNet152,
                self.input_shape, pan_ids=[-33, 80],
                pretrained_weights=str_body_weights)
        elif backbone == "resnet50v2":
            model_body = yolo_keras_app_body(ResNet50V2,
                self.input_shape, pan_ids=[143, 75],
                pretrained_weights=str_body_weights)
        elif backbone == "resnet101v2":
            model_body = yolo_keras_app_body(ResNet101V2,
                self.input_shape, pan_ids=[143, 75],
                pretrained_weights=str_body_weights)
        elif backbone == "resnet152v2":
            model_body = yolo_keras_app_body(ResNet152V2,
                self.input_shape, pan_ids=[143, 75],
                pretrained_weights=str_body_weights)
        else:
            raise ValueError(f"Invalid backbone: {backbone}")

        if pretrained_body is not None:
            model_body.set_weights(pretrained_body.get_weights())

        self._model = yolo_head(
            model_body, self.class_num, anchors)

        if pretrained_weights is not None:
            self._model.load_weights(pretrained_weights)
            if use_arg_anchors:
                self.anchors = anchors
                print("The saved model is loaded and will use the "
                      "argument `anchors` instead of the original anchors.")

        self.grid_shape = self._model.output[0].shape[1:3]

    def read_file_to_dataset(
        self, img_path=None, label_path=None,
        label_format="labelimg",
        rescale=1/255,
        preprocessing=None,
        shuffle=True,
        seed=None,
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
            shuffle: Boolean, default: True.
            seed: An integer, random seed, default: None.
            encoding: A string,
                encoding format of file,
                default: "big5".
            thread_num: An integer,
                specifying the number of threads to read files.

        Returns:
            A tuple: (img: ndarray, label_list: list),
            label_list contains the label of all FPN layers.
            - shape of img: (batch size, img height, img width, channels)
            - shape of label: (batch size, grid height, grid width, channels)
        """
        grid_amp = 2**(self.pan_layers - 1)
        grid_shape = (self.grid_shape[0]*grid_amp,
                      self.grid_shape[1]*grid_amp)

        seq = tools.YoloDataSequence(
            img_path=img_path,
            label_path=label_path,
            label_format=label_format,
            size=self.input_shape[:2],
            rescale=rescale,
            preprocessing=preprocessing,
            grid_shape=grid_shape,
            class_names=self.class_names,
            shuffle=shuffle,
            seed=seed,
            encoding=encoding,
            thread_num=thread_num,
            show_progress=True)
        self._file_names = seq.path_list
        seq.batch_size = len(seq.path_list)

        img_data, label_data = seq[0]

        label_list = [label_data]
        for _ in range(self.pan_layers - 1):
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
                Sequence[i]: (img: ndarray, label_list: list),
            label_list contains the label of all FPN layers.
            - shape of img: (batch size, img height, img width, channels)
            - shape of label: (batch size, grid height, grid width, channels)
        """
        grid_amp = 2**(self.pan_layers - 1)
        grid_shape = (self.grid_shape[0]*grid_amp,
                      self.grid_shape[1]*grid_amp)
        seq = tools.YoloDataSequence(
            img_path=img_path,
            label_path=label_path,
            batch_size=batch_size,
            label_format=label_format,
            size=self.input_shape[:2],
            rescale=rescale,
            preprocessing=preprocessing,
            grid_shape=grid_shape,
            class_names=self.class_names,
            augmenter=augmenter,
            shuffle=shuffle,
            seed=seed,
            encoding=encoding,
            thread_num=thread_num)
        self._file_names = seq.path_list

        v4seq = _Yolov4DataSequence(seq, self.pan_layers)

        return v4seq

    def vis_img(self, img, *label_datas,
                conf_threshold=0.5,
                show_conf=True,
                nms_mode=0,
                nms_threshold=0.45,
                nms_sigma=0.5,
                **kwargs):
        """Visualize the images and annotaions by pyplot.

        Args:
            img: A ndarray of shape(img height, img width, channels).
            *label_datas: Ndarrays,
                shape: (grid height, grid width, channels).
                Multiple label data can be given at once.
            conf_threshold: A float,
                threshold for quantizing output.
            show_conf: A boolean, whether to show confidence score.
            nms_mode: An integer,
                0: Not use NMS.
                1: Use NMS.
                2: Use Soft-NMS.
                3: Use DIoU-NMS.
            nms_threshold: A float,
                threshold for eliminating duplicate boxes.
            nms_sigma: A float,
                sigma for Soft-NMS.
            figsize: (float, float), optional, default: None
                width, height in inches. If not provided, defaults to [6.4, 4.8].
            dpi: float, default: rcParams["figure.dpi"] (default: 100.0)
                The resolution of the figure in dots-per-inch.
                Set as 1.325, then 1 inch will be 1 dot. 
            axis: bool or str
                If a bool, turns axis lines and labels on or off.
                If a string, possible values are:
                https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.axis.html
            savefig_path: None, string, pathLike or file-like object.
            fig_ax: (matplotlib.pyplot.figure, matplotlib.pyplot.axes),
                use this argument to connect visualization of different labels,
                e.g., plot ground truth and prediction label in an image.
            return_fig_ax: A boolean, whether to return fig_ax.
                if it's True, the function won't plot the result this time.
            point_radius: 5.
            point_color: A string or list, defalut: "r".
            box_linewidth: 2.
            box_color: A string or list, defalut: "auto".
            text_color: A string or list, defalut: "w".
            text_padcolor: A string or list, defalut: "auto".
            text_fontsize: 12.
        """
        return tools.vis_img(
            img,
            *label_datas,
            class_names=self.class_names,
            conf_threshold=conf_threshold,
            show_conf=show_conf,
            nms_mode=nms_mode,
            nms_threshold=nms_threshold,
            nms_sigma=nms_sigma,
            version=4,
            **kwargs)

    def loss(self,
             binary_weight=1,
             loss_weight=[1, 5, 1],
             wh_reg_weight=0.01,
             ignore_thresh=0.6,
             truth_thresh=1.0,
             label_smooth=0.0,
             focal_loss_gamma=2):
        """Loss of yolo.

        Args:
            binary_weight: A float or a list of float,
                the ratio of positive and negative for each PAN layers.
            loss_weight: A dictionary or list of length 3,
                specifying the weights of each loss.
                Example:{"box":1, "conf":5, "prob":1},
                or [1, 5, 1].
            wh_reg_weight: A float,
                weight of regularizer that prevents extremely high (t_w, t_h).
            ignore_thresh: A float,
                threshold of ignoring the false positive.
            truth_thresh: A float,
                using multiple anchors for a single
                ground truth: IoU(truth, anchor) > truth_thresh.
            label_smooth: A float,
                using class label smoothing.
            focal_loss_gamma: A integer.

        Returns:
            A loss function conforming to tf.keras specification.
        """
        if (not isinstance(binary_weight, Iterable)
            or len(binary_weight) != self.pan_layers):
            binary_weight = [binary_weight]*self.pan_layers

        if isinstance(loss_weight, dict):
            loss_weight_list = []
            loss_weight_list.append(loss_weight["box"])
            loss_weight_list.append(loss_weight["conf"])
            loss_weight_list.append(loss_weight["prob"])
            loss_weight = loss_weight_list

        loss_list = []
        for pan_id in range(self.pan_layers):
            grid_amp = 2**(pan_id)
            grid_shape = (self.grid_shape[0]*grid_amp,
                          self.grid_shape[1]*grid_amp)
            anchors_id = self.abox_num*pan_id
            loss_list.append(wrap_yolo_loss(
                grid_shape=grid_shape,
                bbox_num=self.abox_num,
                class_num=self.class_num,
                anchors=self.anchors[
                    anchors_id:anchors_id + self.abox_num],
                binary_weight=binary_weight[pan_id],
                loss_weight=loss_weight,
                wh_reg_weight=wh_reg_weight,
                ignore_thresh=ignore_thresh,
                truth_thresh=truth_thresh,
                label_smooth=label_smooth,
                focal_loss_gamma=focal_loss_gamma))
        return loss_list

    def metrics(self, kind="obj_acc"):
        """Metrics of yolo.

        Args:
            kind: A string,
                one of "obj_acc", "mean_iou", "class_acc" or "recall",
                use "obj_acc+mean_iou", "mean_iou+recall0.6"
                to specify multiple metrics.
                The number after "recall" indicates the iou threshold.

        Returns:
            A list of metric function conforming to tf.keras specification.
        """
        metrics_list = [[] for _ in range(self.pan_layers)]
        for pan_id in range(self.pan_layers):
            grid_amp = 2**(pan_id)
            grid_shape = (self.grid_shape[0]*grid_amp,
                            self.grid_shape[1]*grid_amp)

            if "obj" in kind:
                metrics_list[pan_id].append(
                    wrap_obj_acc(
                        grid_shape,
                        self.abox_num,
                        self.class_num))
            if "iou" in kind:
                metrics_list[pan_id].append(
                    wrap_mean_iou(
                        grid_shape,
                        self.abox_num,
                        self.class_num))
            if "class" in kind:
                metrics_list[pan_id].append(
                    wrap_class_acc(
                        grid_shape,
                        self.abox_num,
                        self.class_num))
            if "recall" in kind:
                iou_threshold = kind[kind.find("recall") + 6:]
                end = iou_threshold.rfind("+")
                if end < 0:
                    end = None
                iou_threshold = iou_threshold[:end]
                if iou_threshold == "":
                    iou_threshold = 0.5
                else:
                    iou_threshold = float(iou_threshold)

                metrics_list[pan_id].append(
                    wrap_recall(
                        grid_shape,
                        self.abox_num,
                        self.class_num,
                        iou_threshold=iou_threshold))
        return metrics_list
