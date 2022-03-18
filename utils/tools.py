# Copyright 2020 Samson. All Rights Reserved.
# =============================================================================

"""Utilities and tools for Yolo.
"""

from PIL import Image
import base64
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle, BoxStyle
from bs4 import BeautifulSoup
import json
from io import BytesIO
import base64
from math import ceil
import threading
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from numpy.core.fromnumeric import argmax
from tensorflow.keras.utils import Sequence
import pandas as pd
import xml.etree.ElementTree as ET

epsilon = 1e-07


def read_img(path, size=(512, 512), rescale=None):
    """Read images as ndarray.

    Args:
        size: A tuple of 2 integers,
            shape(heights, widths).
        rescale: A float or None,
            specifying how the image value should be scaled.
            If None, no scaled.
    """
    img_list = [f for f in os.listdir(path) if not f.startswith(".")]
    data = np.empty((len(img_list), *size, 3))
    size = size[1], size[0]
    
    for i, _path in enumerate(img_list):
        img = Image.open(path + os.sep + _path)
        img = img.resize(size)
        img = img.convert("RGB")  
        img = np.array(img)
        if rescale is not None:
            img = img*rescale
        data[i] = img
     
    return data


def _process_img(img, size):
    size = size[1], size[0]
    zoom_r = np.array(img.size)/np.array(size)
    img = img.resize(size)
    img = img.convert("RGB")
    img = np.array(img)
    return img, zoom_r


def read_file(img_path=None,
              label_path=None,
              label_format="labelimg",
              size=(448, 448),
              grid_shape=(7, 7),
              class_names=[],
              rescale=1/255,
              preprocessing=None,
              augmenter=None,
              aug_times=1,
              shuffle=True,
              seed=None,
              encoding="big5",
              thread_num=10):
    """Read the images and annotations created by labelimg or labelme.

    Args:
        img_path: A string, 
            file path of images.
        label_path: A string,
            file path of annotations.
        label_format: A string,
            one of "labelimg" and "labelme".
        size: A tuple of 2 integer,
            shape of output image(heights, widths).
        grid_shape: A tuple or list of integers(heights, widths), 
            specifying input images will be divided into 
            grid_shape[0] x grid_shape[1] grids.
        class_names: A list of string,
            containing all label names.
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
        shape of img: (sample_num, img_heights, img_widths, channels)
        shape of label: (sample_num, grid_heights, grid_widths, info)
    """
    def _process_paths(path_list, aug_times):
        path_list = np.array(path_list)
        U_num = path_list.dtype.itemsize//4 + 5
        dtype = "<U" + str(U_num)
        filepaths = np.empty((len(path_list),
                             aug_times),
                             dtype = dtype)
        filepaths[:, 0] = path_list
        filepaths[:, 1:] = np.char.add(filepaths[:, 0:1], "(aug)")
        path_list = filepaths.flatten()
        return path_list
    
    def _encode_to_array(img, bbs,
                         grid_shape, pos, name, labels):
        img_data[pos] = img

        grid_height = img.shape[0]/grid_shape[0]
        grid_width = img.shape[1]/grid_shape[1]
        img_height = img.shape[0]
        img_width = img.shape[1]

        for label_i, box in enumerate(bbs.bounding_boxes):
            x = (box.x1 + (box.x2-box.x1)/2)
            y = (box.y1 + (box.y2-box.y1)/2)
            w = (box.x2-box.x1)
            h = (box.y2-box.y1)

            x_i = int(x//grid_width) #Grid x coordinate
            y_i = int(y//grid_height) #Grid y coordinate

            if x_i < grid_shape[1] and y_i < grid_shape[0]:
                if label_data[pos, y_i, x_i, 4] == 1: 
                    #Detect whether objects overlap
                    print("Notice! Repeat!:", name, y_i, x_i)

                label_data[pos, y_i, x_i, 0] = (
                    x%grid_width/grid_width)
                label_data[pos, y_i, x_i, 1] = (
                    y%grid_height/grid_height)
                label_data[pos, y_i, x_i, 2] = (
                    w/img_width)
                label_data[pos, y_i, x_i, 3] = (
                    h/img_height)
                label_data[pos, y_i, x_i, 4] = 1
                label_data[pos, y_i, x_i, 5 + labels[label_i]] = 1

    def _imgaug_to_array(img, bbs,
            grid_shape, pos, name, labels):
        _encode_to_array(img, bbs,
            grid_shape, pos, name, labels)
        if augmenter is not None:
            for aug_i in range(1, aug_times):
                img_aug, bbs_aug = augmenter(image=img,
                                             bounding_boxes=bbs)
                _encode_to_array(img_aug, bbs_aug,
                    grid_shape, pos + aug_i, name, labels)

    def _read_labelimg(_path_list, _pos):
        for _path_list_i, name in enumerate(_path_list):
            pos = (_pos + _path_list_i)*aug_times

            with open(os.path.join(
                      label_path,
                      name[:name.rfind(".")] + ".xml"),
                      encoding=encoding) as f:
                soup = BeautifulSoup(f.read(), "xml")

            img = Image.open(os.path.join(img_path, name))
            img, zoom_r = _process_img(img, size)

            bbs = []
            labels = []
            for obj in soup.select("object"):
                if obj.select_one("name").text in class_names:
                    label = class_names.index(obj.select_one("name").text)
                    labels.append(label)
                    xmin = int(obj.select_one("xmin").text)/zoom_r[0]
                    xmax = int(obj.select_one("xmax").text)/zoom_r[0]
                    ymin = int(obj.select_one("ymin").text)/zoom_r[1]
                    ymax = int(obj.select_one("ymax").text)/zoom_r[1]

                    bbs.append(BoundingBox(x1=xmin,
                                           y1=ymin,
                                           x2=xmax,
                                           y2=ymax))
            bbs = BoundingBoxesOnImage(bbs, shape=img.shape)
            _imgaug_to_array(img, bbs,
                grid_shape, pos, name, labels)

    def _read_labelme(_path_list, _pos):
        for _path_list_i, name in enumerate(_path_list):
            pos = (_pos + _path_list_i)*aug_times

            with open(os.path.join(
                      label_path,
                      name[:name.rfind(".")] + ".json"),
                      encoding=encoding) as f:
                jdata = f.read()
                data = json.loads(jdata)

            if img_path is None:
                img64 = data['imageData']
                img = Image.open(BytesIO(base64.b64decode(img64)))
            else:
                img = Image.open(os.path.join(img_path, name))

            img, zoom_r = _process_img(img, size)

            bbs = []
            labels = []
            data_shapes = data['shapes']
            for i in range(len(data_shapes)):
                label_text = data_shapes[i]['label']
                if (data_shapes[i]['shape_type'] == 'rectangle'
                        and label_text in class_names):
                    label = class_names.index(label_text)
                    labels.append(label)
                    point = np.array(data['shapes'][i]['points'])
                    point = point/zoom_r

                    bbs.append(BoundingBox(x1=point[0, 0],
                                           y1=point[0, 1],
                                           x2=point[1, 0],
                                           y2=point[1, 1]))
            bbs = BoundingBoxesOnImage(bbs, shape=img.shape)
            _imgaug_to_array(img, bbs,
                grid_shape, pos, name, labels)
    if (label_format == "labelme" 
        and (img_path is None or label_path is None)):
        if label_path is None:
            label_path = img_path
            img_path = None
        path_list = os.listdir(label_path)
        path_list = [f for f in path_list if f.endswith(".json")]
    else:
        path_list = os.listdir(img_path)
        path_list = [f for f in path_list if not f.startswith(".")]
    path_list_len = len(path_list)
    
    class_num = len(class_names)
    if augmenter is None:
        aug_times = 1

    img_data = np.empty((path_list_len*aug_times,
                           *size, 3))
    label_data = np.zeros((path_list_len*aug_times,
                           *grid_shape,
                           5 + class_num))

    threads = []
    workers_num = ceil(path_list_len/thread_num)
    if label_format == "labelimg":
        thread_func = _read_labelimg
    elif label_format == "labelme":
        thread_func = _read_labelme
    else:
        raise ValueError("Invalid format: %s" % label_format)  

    for path_list_i in range(0, path_list_len, workers_num):
        threads.append(
            threading.Thread(target = thread_func,
            args = (
                path_list[path_list_i:path_list_i + workers_num],
                path_list_i))
        )
    for thread in threads:
        thread.start()                
    for thread in threads:
        thread.join()

    if rescale is not None:
        img_data = img_data*rescale
    if preprocessing is not None:
        img_data = preprocessing(img_data)

    path_list = _process_paths(path_list, aug_times)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        shuffle_index = np.arange(len(img_data))
        np.random.shuffle(shuffle_index)
        img_data = img_data[shuffle_index]
        label_data = label_data[shuffle_index]
        path_list = path_list[shuffle_index]

    path_list = path_list.tolist()

    return img_data, label_data, path_list


class YoloDataSequence(Sequence):
    """Read the images and annotations
    created by labelimg or labelme as a Sequence.

    Args:
        img_path: A string, 
            file path of images.
        label_path: A string,
            file path of annotations.
        batch_size:  An integer,
            size of the batches of data (default: 20).
        label_format: A string,
            one of "labelimg" and "labelme".
        size: A tuple of 2 integer,
            shape of output image(heights, widths).
        rescale: A float or None,
            specifying how the image value should be scaled.
            If None, no scaled.
        preprocessing: A function of data preprocessing,
            (e.g. noralization, shape manipulation, etc.)
        grid_shape: A tuple or list of integers(heights, widths), 
            specifying input images will be divided into 
            grid_shape[0] x grid_shape[1] grids.
        class_names: A list of string,
            containing all label names.
        augmenter: A `imgaug.augmenters.meta.Sequential` instance.
        shuffle: Boolean, default: True.
        seed: An integer, random seed, default: None.
        encoding: A string,
            encoding format of file,
            default: "big5".
        thread_num: An integer,
            specifying the number of threads to read files.

    Returns:
        A tf.Sequence.
    """
    def __init__(self, img_path=None,
                 label_path=None,
                 batch_size=20,
                 label_format="labelimg",
                 size=(448, 448),
                 rescale=1/255,
                 preprocessing=None,
                 grid_shape=(7, 7),
                 class_names=[],
                 augmenter=None,
                 shuffle=True,
                 seed=None,
                 encoding="big5",
                 thread_num=1):
        self.img_path = img_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.label_format = label_format
        self.size = size
        self.rescale = rescale
        self.preprocessing = preprocessing
        self.grid_shape = grid_shape
        self.class_names = class_names
        self.class_num = len(class_names)
        self.augmenter = augmenter
        self.shuffle = shuffle
        self.seed = seed
        self.encoding = encoding
        self.thread_num = thread_num

        if (label_format == "labelme" 
            and (img_path is None or label_path is None)):
            if label_path is None:
                self.label_path = img_path
                self.img_path = None
            path_list = os.listdir(self.label_path)
            self.path_list = [f for f in path_list if f.endswith(".json")]
        else:
            path_list = os.listdir(img_path)
            self.path_list = [f for f in path_list if not f.startswith(".")]

        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            self.path_list = np.array(self.path_list)
            np.random.shuffle(self.path_list)
            self.path_list = self.path_list.tolist()

    def __len__(self):
        return ceil(len(self.path_list)/self.batch_size)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Sequence index out of range")
        def _encode_to_array(img, bbs,
                             grid_shape, pos, labels):
            img_data[pos] = img

            grid_height = img.shape[0]/grid_shape[0]
            grid_width = img.shape[1]/grid_shape[1]
            img_height = img.shape[0]
            img_width = img.shape[1]

            for label_i, box in enumerate(bbs.bounding_boxes):
                x = (box.x1 + (box.x2-box.x1)/2)
                y = (box.y1 + (box.y2-box.y1)/2)
                w = (box.x2-box.x1)
                h = (box.y2-box.y1)

                x_i = int(x//grid_width) #Grid x coordinate
                y_i = int(y//grid_height) #Grid y coordinate

                if x_i < grid_shape[1] and y_i < grid_shape[0]:
                    label_data[pos, y_i, x_i, 0] = (
                        x%grid_width/grid_width)
                    label_data[pos, y_i, x_i, 1] = (
                        y%grid_height/grid_height)
                    label_data[pos, y_i, x_i, 2] = (
                        w/img_width)
                    label_data[pos, y_i, x_i, 3] = (
                        h/img_height)
                    label_data[pos, y_i, x_i, 4] = 1
                    label_data[pos, y_i, x_i, 5 + labels[label_i]] = 1
        
        def _imgaug_to_array(img, bbs,
                grid_shape, pos, labels):
            if self.augmenter is None:
                _encode_to_array(img, bbs,
                    grid_shape, pos, labels)
            else:
                img_aug, bbs_aug = self.augmenter(
                    image=img,
                    bounding_boxes=bbs)
                _encode_to_array(img_aug, bbs_aug,
                    grid_shape, pos, labels)

        def _read_labelimg(_path_list, _pos):
            for i, name in enumerate(_path_list):
                pos = (_pos + i)
                with open(os.path.join(
                          self.label_path,
                          name[:name.rfind(".")] + ".xml"),
                          encoding=self.encoding) as f:
                    soup = BeautifulSoup(f.read(), "xml")

                img = Image.open(os.path.join(self.img_path, name))
                img, zoom_r = _process_img(img, self.size)

                bbs = []
                labels = []
                for obj in soup.select("object"):
                    if obj.select_one("name").text in self.class_names:
                        label_text = obj.select_one("name").text
                        label = self.class_names.index(label_text)
                        labels.append(label)
                        xmin = int(obj.select_one("xmin").text)/zoom_r[0]
                        xmax = int(obj.select_one("xmax").text)/zoom_r[0]
                        ymin = int(obj.select_one("ymin").text)/zoom_r[1]
                        ymax = int(obj.select_one("ymax").text)/zoom_r[1]

                        bbs.append(BoundingBox(x1=xmin,
                                               y1=ymin,
                                               x2=xmax,
                                               y2=ymax))
                bbs = BoundingBoxesOnImage(bbs, shape=img.shape)
                _imgaug_to_array(img, bbs,
                    self.grid_shape, pos, labels)

        def _read_labelme(_path_list, _pos):
            for i, name in enumerate(_path_list):
                pos = (_pos + i)
                with open(os.path.join(
                          self.label_path,
                          name[:name.rfind(".")] + ".json"),
                          encoding=self.encoding) as f:
                    jdata = f.read()
                    data = json.loads(jdata)

                if self.img_path is None:
                    img64 = data['imageData']
                    img = Image.open(BytesIO(base64.b64decode(img64)))
                else:
                    img = Image.open(os.path.join(self.img_path, name))

                img, zoom_r = _process_img(img, self.size)

                bbs = []
                labels = []
                data_shapes = data['shapes']
                for i in range(len(data_shapes)):
                    label_text = data_shapes[i]['label']
                    if (data_shapes[i]['shape_type'] == 'rectangle'
                            and label_text in self.class_names):
                        label = self.class_names.index(label_text)
                        labels.append(label)
                        point = np.array(data['shapes'][i]['points'])
                        point = point/zoom_r

                        bbs.append(BoundingBox(x1=point[0, 0],
                                               y1=point[0, 1],
                                               x2=point[1, 0],
                                               y2=point[1, 1]))
                bbs = BoundingBoxesOnImage(bbs, shape=img.shape)
                _imgaug_to_array(img, bbs,
                    self.grid_shape, pos, labels)

        total_len = len(self.path_list)
        if (idx + 1)*self.batch_size > total_len:
            batch_size = total_len % self.batch_size
        else:
            batch_size = self.batch_size
        img_data = np.empty((batch_size, *self.size, 3))
        label_data = np.zeros((batch_size,
                               *self.grid_shape,
                               5 + self.class_num))
        start_idx = idx*self.batch_size
        end_idx = (idx + 1)*self.batch_size
        path_list = self.path_list[start_idx:end_idx]
        if self.label_format == "labelimg":
            thread_func = _read_labelimg
        elif self.label_format == "labelme":
            thread_func = _read_labelme
        else:
            raise ValueError("Invalid format: %s" % self.label_format)

        threads = []
        workers = ceil(len(path_list)/self.thread_num)

        for worker_i in range(0, len(path_list), workers):
            threads.append(
                threading.Thread(target=thread_func,
                args=(path_list[worker_i : worker_i+workers],
                      worker_i)))
        for thread in threads:
            thread.start()                
        for thread in threads:
            thread.join()

        if self.rescale is not None:
            img_data = img_data*self.rescale
        if self.preprocessing is not None:
            img_data = self.preprocessing(img_data)
      
        return img_data, label_data


def down2xlabel(label_data):
    batches = label_data.shape[0]
    grid_h = label_data.shape[1]
    grid_w = label_data.shape[2]
    channels = label_data.shape[3]

    new_label = np.zeros((
        batches,
        grid_h//2,
        grid_w//2,
        channels))

    for batch in range(batches):
        for i in range(0, grid_h, 2):
            for j in range(0, grid_w, 2):
                crop = label_data[batch][i:i+2, j:j+2]
                if crop[..., 4].max() == 1:
                    max_id = (crop[..., 2]*crop[..., 3]).argmax()
                    crop = crop[max_id//2, max_id%2]
                    crop_xy = crop[:2]
                    crop_xy = (crop_xy + [max_id%2, max_id//2])/2
                    crop_whcp = crop[2:]
                    new_label[batch][i//2, j//2, :2] = crop_xy
                    new_label[batch][i//2, j//2, 2:] = crop_whcp
    return new_label


def decode(*label_datas,
           class_num=1,
           threshold=0.5,
           version=1):
    """Decode the prediction from yolo model.

    Args:
        *label_datas: Ndarrays,
            shape: (grid_heights, grid_widths, info).
            Multiple label data can be given at once.
        class_num:  An integer,
            number of classes.
        threshold: A float,
            threshold for quantizing output.
        version: An integer,
            specifying the decode method, yolov1、v2 or v3.

    Return:
        Numpy.ndarray with shape: (N, 7).
            7 values represent:
            x, y, w, h, c, class index, class probability.
    """
    output = []
    for label_data in label_datas:
        grid_shape = label_data.shape[:2]
        if version == 1:
            bbox_num = (label_data.shape[-1] - class_num)//5
            xywhc = np.reshape(label_data[..., :-class_num],
                               (*grid_shape, bbox_num, 5))
            prob = np.expand_dims(
                label_data[..., -class_num:], axis=-2)
        elif version == 2 or version == 3:
            bbox_num = label_data.shape[-1]//(5 + class_num)
            label_data = np.reshape(label_data,
                                    (*grid_shape,
                                     bbox_num, 5 + class_num))
            xywhc = label_data[..., :5]
            prob = label_data[..., -class_num:]
        else:
            raise ValueError("Invalid version: %s" % version)   

        joint_conf = xywhc[..., 4:5]*prob
        where = np.where(joint_conf >= threshold)

        for i in range(len(where[0])):
            x_i = where[1][i]
            y_i = where[0][i]
            box_i = where[2][i]
            class_i = where[3][i]

            x_reg = xywhc[y_i, x_i, box_i, 0]
            y_reg = xywhc[y_i, x_i, box_i, 1]
            w_reg = xywhc[y_i, x_i, box_i, 2]
            h_reg = xywhc[y_i, x_i, box_i, 3]
            conf = xywhc[y_i, x_i, box_i, 4]

            x = (x_i + x_reg)/grid_shape[1]
            y = (y_i + y_reg)/grid_shape[0]
            
            w = w_reg
            h = h_reg
            
            if version == 1:
                p = prob[y_i, x_i, 0, class_i]
            else:
                p = prob[y_i, x_i, box_i, class_i]
            output.append([x, y, w, h, conf, class_i, p])
    output = np.array(output, dtype="float")
    return output

def vis_img(img,
            *label_datas,
            class_names=[],
            conf_threshold=0.5,
            show_conf=True,
            nms_mode=0,
            nms_threshold=0.5,
            nms_sigma=0.5,
            version=1, 
            figsize=None,
            dpi=None,
            axis="off",
            savefig_path=None,
            fig_ax=None,
            return_fig_ax=False,
            point_radius=5,
            point_color="r",
            box_linewidth=2,
            box_color="auto",
            text_color="w",
            text_padcolor="auto",
            text_fontsize=12):
    """Visualize the images and annotaions by pyplot.

    Args:
        img: A ndarray of shape(img_heights, img_widths, channels).
        *label_datas: Ndarrays,
            shape: (grid_heights, grid_widths, info).
            Multiple label data can be given at once.
        class_names: A list of string,
            containing all label names.
        conf_threshold: A float,
            threshold for quantizing output.
        show_conf: A boolean, whether to show confidence score.
        nms_mode: An integer,
            0: Not use NMS.
            1: Use NMS.
            2: Use Soft-NMS.
        nms_threshold: A float,
            threshold for eliminating duplicate boxes.
        nms_sigma: A float,
            sigma for Soft-NMS.
        version: An integer,
            specifying the decode method, yolov1、v2 or v3.
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
    class_num = len(class_names)

    if isinstance(point_color, str):
        point_color = [point_color]*class_num
    if box_color == "auto":
        box_color = point_color
    if text_padcolor == "auto":
        text_padcolor = point_color
    if isinstance(box_color, str):
        box_color = [box_color]*class_num
    if isinstance(text_color, str):
        text_color = [text_color]*class_num
    if isinstance(text_padcolor, str):
        text_padcolor = [text_padcolor]*class_num


    nimg = np.copy(img)

    xywhcp = decode(*label_datas,
                    class_num=class_num,
                    threshold=conf_threshold,
                    version=version)
    if nms_mode > 0 and len(xywhcp) > 0:
        if nms_mode == 1:
            xywhcp = nms(xywhcp, nms_threshold)
        elif nms_mode == 2:
            xywhcp = soft_nms(
                xywhcp, nms_threshold,
                conf_threshold, nms_sigma)

    xywhc = xywhcp[..., :5]
    prob = xywhcp[..., 5:] 

    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
        ax.imshow(img)
        ax.axis(axis)

    for i in range(len(xywhc)):
        x = xywhc[i][0]*nimg.shape[1]
        y = xywhc[i][1]*nimg.shape[0]
        
        w = xywhc[i][2]*nimg.shape[1]
        h = xywhc[i][3]*nimg.shape[0]

        class_i = int(prob[i][0])
        label = class_names[class_i]

        point_min = int(x - w/2), int(y - h/2)
        # point_max = int(x + w/2), int(y + h/2)

        cir = Circle((x,y),
                     radius=point_radius,
                     color=point_color[class_i])
        
        rect = Rectangle(point_min,
                         w, h,
                         linewidth=box_linewidth,
                         edgecolor=box_color[class_i],
                         facecolor="none")
        if show_conf:
            conf = xywhc[i][4]*prob[i][1]
            text = "%s:%.2f" % (label, conf)
        else:
            text = label
        if text_fontsize > 0:
            ax.text(*point_min,
                    text,
                    color=text_color[class_i],
                    bbox=dict(boxstyle=BoxStyle.Square(pad=0.2),
                              color=text_padcolor[class_i]),
                    fontsize=text_fontsize,
                    )

        ax.add_patch(cir)
        ax.add_patch(rect)
    if savefig_path is not None:
        fig.savefig(savefig_path, bbox_inches='tight', pad_inches = 0)

    if return_fig_ax:
        return fig, ax
    else:
        plt.show()


def get_class_weight(label_data, method="alpha"):
    """Get the weight of the category.

    Args:
        label_data: A ndarray,
        shape: (batch_size, grid_heights, grid_widths, info).
        method: A string,
            one of "alpha"、"log"、"effective"、"binary".

    Returns:
        A list containing the weight of each category.
    """
    class_weight = []
    if method != "alpha":
        total = 1
        for i in label_data.shape[:-1]:
            total *= i
        if method == "effective":
            beta = (total - 1)/total
    for i in range(label_data.shape[-1]):
        samples_per_class = label_data[..., i].sum()
        if method == "effective":
            effective_num = 1 - np.power(beta, samples_per_class)
            class_weight.append((1 - beta)/effective_num)
        elif method == "binary":
            class_weight.append(samples_per_class/(total - samples_per_class))
        else:
            class_weight.append(1/samples_per_class)
    class_weight = np.array(class_weight)
    if method == "log":
        class_weight = np.log(total*class_weight)
 
    if method != "binary":
        class_weight = class_weight/np.sum(class_weight)*len(class_weight)

    return class_weight


def cal_iou(xywh_true, xywh_pred):
    """Calculate IOU of two tensors.

    Args:
        xywh_true: A tensor or array-like of shape (..., 4).
            (x, y) should be normalized by image size.
        xywh_pred: A tensor or array-like of shape (..., 4).
    Returns:
        An iou_scores array.
    """
    xy_true = xywh_true[..., 0:2] # N*1*1*1*(S*S)*2
    wh_true = xywh_true[..., 2:4]

    xy_pred = xywh_pred[..., 0:2] # N*S*S*B*1*2
    wh_pred = xywh_pred[..., 2:4]
    
    half_xy_true = wh_true / 2.
    mins_true    = xy_true - half_xy_true
    maxes_true   = xy_true + half_xy_true

    half_xy_pred = wh_pred / 2.
    mins_pred    = xy_pred - half_xy_pred
    maxes_pred   = xy_pred + half_xy_pred       
    
    intersect_mins  = np.maximum(mins_pred,  mins_true)
    intersect_maxes = np.minimum(maxes_pred, maxes_true)
    intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = wh_true[..., 0] * wh_true[..., 1]
    pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = intersect_areas/(union_areas + epsilon)
    
    return iou_scores
 

def nms(xywhcp, nms_threshold=0.5):
    """Non-Maximum Suppression.

    Args:
        xywhcp: output from `decode()`.
        nms_threshold: A float, default is 0.5.

    Returns:
        xywhcp through nms.
    """
    prob = xywhcp[..., 5:]
    argmax_prob = prob.argmax(axis=-1)

    xywhcp_new = []
    for i_class in range(prob.shape[-1]):
        xywhcp_class = xywhcp[argmax_prob==i_class]
        xywhc_class = xywhcp_class[..., :5]
        prob_class = xywhcp_class[..., 5 + i_class]

        xywhc_axis0 = np.reshape(
            xywhc_class, (-1, 1, 5))
        xywhc_axis1 = np.reshape(
            xywhc_class, (1, -1, 5))

        iou_scores = cal_iou(xywhc_axis0, xywhc_axis1)
        conf = xywhc_class[..., 4]*prob_class
        sort_index = np.argsort(conf)[::-1]

        white_list = []
        delete_list = []
        for conf_index in sort_index:
            white_list.append(conf_index)
            if conf_index not in delete_list:
                iou_score = iou_scores[conf_index]
                overlap_indexes = np.where(iou_score >= nms_threshold)[0]

                for overlap_index in overlap_indexes:
                    if overlap_index not in white_list:
                        delete_list.append(overlap_index)
        xywhcp_class = np.delete(xywhcp_class, delete_list, axis=0)
        xywhcp_new.append(xywhcp_class)
    xywhcp = np.vstack(xywhcp_new)
    return xywhcp


def soft_nms(xywhcp, nms_threshold=0.5, conf_threshold=0.5, sigma=0.5):
    """Soft Non-Maximum Suppression.

    Args:
        xywhcp: output from `decode()`.
        nms_threshold: A float, default is 0.5.
        conf_threshold: A float,
            threshold for quantizing output.
        sigma: A float,
            sigma for Soft NMS.

    Returns:
        xywhcp through nms.
    """
    prob = xywhcp[..., 5:]
    argmax_prob = prob.argmax(axis=-1)

    xywhcp_new = []
    for i_class in range(prob.shape[-1]):
        xywhcp_class = xywhcp[argmax_prob==i_class]
        xywhc_class = xywhcp_class[..., :5]
        prob_class = xywhcp_class[..., 5 + i_class]

        xywhc_axis0 = np.reshape(
            xywhc_class, (-1, 1, 5))
        xywhc_axis1 = np.reshape(
            xywhc_class, (1, -1, 5))

        iou_scores = cal_iou(xywhc_axis0, xywhc_axis1)
        conf = xywhc_class[..., 4]*prob_class
        sort_index = np.argsort(conf)[::-1]

        white_list = []
        delete_list = []
        for conf_index in sort_index:
            white_list.append(conf_index)
            iou_score = iou_scores[conf_index]
            overlap_indexes = np.where(iou_score >= nms_threshold)[0]

            for overlap_index in overlap_indexes:
                if overlap_index not in white_list:
                    conf_decay = np.exp(-1*(iou_score[overlap_index]**2)/sigma)
                    conf[overlap_index] *= conf_decay
                    if conf[overlap_index] < conf_threshold:
                        delete_list.append(overlap_index)
        xywhcp_class = np.delete(xywhcp_class, delete_list, axis=0)
        xywhcp_new.append(xywhcp_class)
    xywhcp = np.vstack(xywhcp_new)
    return xywhcp


def create_score_mat(*args, **kwargs):
    """The location of this function has been changed.
    Import it using `from utils.measurement import create_score_mat`
    """
    raise ImportError(
        "The location of this function has been changed. "
        "Import it using"
        "`from utils.measurement import create_score_mat`"
        )


def array_to_json(path,
                  img_size,                 
                  *label_datas,
                  class_names=[],
                  conf_threshold=0.5,
                  nms_mode=0,
                  nms_threshold=0.5,
                  nms_sigma=0.5,
                  version=3):
    """Convert Yolo output array to json file.

    Args:
        path: A string,
            the path to store the json file.
        img_size: A tuple of 2 integers (heights, widths),
            original size of image.
        *label_datas: Ndarrays,
            shape: (grid_heights, grid_widths, info).
            Multiple label data can be given at once.
        class_names: A list, containing all label names.
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
        version: An integer,
            specifying the decode method, yolov1、v2 or v3.     
    """
    class_num = len(class_names)

    xywhcp = decode(*label_datas,
                    class_num=class_num,
                    threshold=conf_threshold,
                    version=version)
    if nms_mode > 0 and len(xywhcp) > 0:
        if nms_mode == 1:
            xywhcp = nms(xywhcp, nms_threshold)
        elif nms_mode == 2:
            xywhcp = soft_nms(
                xywhcp, nms_threshold,
                conf_threshold, nms_sigma)

    xywhc = xywhcp[..., :5]
    prob = xywhcp[..., 5:] 

    obj_list = []
    for i in range(len(xywhc)):
        x = xywhc[i][0]*img_size[1]
        y = xywhc[i][1]*img_size[0]
        
        w = xywhc[i][2]*img_size[1]
        h = xywhc[i][3]*img_size[0]

        class_i = int(prob[i][0])
        label = class_names[class_i]
        conf = xywhc[i][4]*prob[i][1]

        point_min = [x - w/2, y - h/2]
        point_max = [x + w/2, y + h/2]
        points = [point_min, point_max]


        obj_list.append({"label": label,
                         "points": points,
                         "shape_type": "rectangle",
                         "confidence": conf})

    data = {"shapes": obj_list,
            "imageHeight": img_size[0],
            "imageWidth": img_size[1]}

    with open(path, "w") as fp:
        fp.write(str(data).replace("'", "\""))


def array_to_xml(path,
                 img_size,                 
                 *label_datas,
                 class_names=[],
                 conf_threshold=0.5,
                 nms_mode=0,
                 nms_threshold=0.5,
                 nms_sigma=0.5,
                 version=3):
    """Convert Yolo output array to xml file.

    Args:
        path: A string,
            the path to store the xml file.
        img_size: A tuple of 2 integers (heights, widths),
            original size of image.
        *label_datas: Ndarrays,
            shape: (grid_heights, grid_widths, info).
            Multiple label data can be given at once.
        class_names: A list, containing all label names.
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
        version: An integer,
            specifying the decode method, yolov1、v2 or v3.     
    """
    class_num = len(class_names)

    xywhcp = decode(*label_datas,
                    class_num=class_num,
                    threshold=conf_threshold,
                    version=version)
    if nms_mode > 0 and len(xywhcp) > 0:
        if nms_mode == 1:
            xywhcp = nms(xywhcp, nms_threshold)
        elif nms_mode == 2:
            xywhcp = soft_nms(
                xywhcp, nms_threshold,
                conf_threshold, nms_sigma)

    xywhc = xywhcp[..., :5]
    prob = xywhcp[..., 5:] 

    root = ET.Element("annotation")
    for i in range(len(xywhc)):
        x = xywhc[i][0]*img_size[1]
        y = xywhc[i][1]*img_size[0]
        
        w = xywhc[i][2]*img_size[1]
        h = xywhc[i][3]*img_size[0]

        class_i = int(prob[i][0])
        label = class_names[class_i]
        conf = xywhc[i][4]*prob[i][1]

        object = ET.Element("object")
        root.append(object)

        name = ET.SubElement(object, "name")
        name.text = label

        bndbox = ET.Element("bndbox")
        object.append(bndbox)

        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(x - w/2))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(y - h/2))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(x + w/2))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(y + h/2))

        confidence = ET.SubElement(object, "confidence")
        confidence.text = str(conf)

    tree = ET.ElementTree(root)
    
    with open(path, "wb") as files:
        tree.write(files)