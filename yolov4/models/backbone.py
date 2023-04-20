"""YOLOv4 Model Defined in Keras.
"""

from functools import wraps
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import softplus, tanh
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, Add
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model


class Mish(Layer):
    """
    Mish Activation Function.
    `mish(x) = x * tanh(softplus(x))`
    Examples:
        >>> input_tensor = Input(input_shape)
        >>> output = Mish()(input_tensor)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.trainable = False

    def call(self, inputs):
        return inputs * tanh(softplus(inputs))


class Anchor(Layer):
    """
    YOLO Anchor Layer.
    `Anchor(tensor) = box * exp(tensor)`
        box: [x, y]
    Examples:
        >>> input_tensor = Input(input_shape)
        >>> output = Anchor([0.0197, 0.0263])(input_tensor)
    """
    def __init__(self, box, **kwargs):
        super().__init__(**kwargs)
        self.box = np.array(box)
        self.trainable = False

    def build(self, input_shape):
        self.weight = tf.Variable(
            self.box.reshape((*((1,)*(len(input_shape) - 1)), -1)),
            dtype="float32", trainable=True)

    def call(self, inputs):
        return tf.multiply(tf.exp(inputs), self.weight)


class DarknetConv2D(Conv2D):
    """Convolution2D with Darknet parameters.
    """
    __doc__ += Conv2D.__doc__
    def __init__(self, *args, **kwargs):
        kwargs["kernel_initializer"] = RandomNormal(mean=0.0, stddev=0.02)
        if kwargs.get("strides") == (2, 2):
            kwargs["padding"] = "valid"
        else:
            kwargs["padding"] = "same"
        super().__init__(*args, **kwargs)


def conv2d_bn_leaky(tensor, *args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
    """
    bn_name = None
    acti_name = None
    if "name" in kwargs:
        name = kwargs["name"]
        kwargs["name"] = name + "_conv"
        bn_name = name + "_bn"
        acti_name = name + "_leaky"
    kwargs["use_bias"] = False

    tensor = DarknetConv2D(*args, **kwargs)(tensor)
    tensor = BatchNormalization(name=bn_name)(tensor)
    tensor = LeakyReLU(alpha=0.1, name=acti_name)(tensor)

    return tensor


def conv2d_bn_mish(tensor, *args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and Mish.
    """
    bn_name = None
    acti_name = None
    if "name" in kwargs:
        name = kwargs["name"]
        kwargs["name"] = name + "_conv"
        bn_name = name + "_bn"
        acti_name = name + "_mish"
    kwargs["use_bias"] = False

    tensor = DarknetConv2D(*args, **kwargs)(tensor)
    tensor = BatchNormalization(name=bn_name)(tensor)
    tensor = Mish(name=acti_name)(tensor)

    return tensor


def resblock_module(tensor, mid_filters, out_filters, name="block1"):
    """CSPDarkNet53 residual block module."""
    skip_tensor = tensor
    tensor = conv2d_bn_mish(
        tensor, mid_filters, 1, name=name + "_1x1")
    tensor = conv2d_bn_mish(
        tensor, out_filters, 3, name=name + "_3x3")
    tensor = Add(name=name + "_add")([tensor, skip_tensor])
    return tensor


def resstage_module(tensor, num_filters, num_blocks,
                    is_narrow=True, name="block1"):
    """CSPDarkNet53 residual stage module."""
    mid_filters = num_filters//2 if is_narrow else num_filters

    tensor = ZeroPadding2D(((1, 0), (1, 0)), name=name + "_pad")(tensor)
    tensor = conv2d_bn_mish(
        tensor, num_filters, 3, strides=(2, 2), name=name + "_dn")
    cross_tensor = conv2d_bn_mish(
        tensor, mid_filters, 1, name=name + "_cross")
    tensor = conv2d_bn_mish(
        tensor, mid_filters, 1, name=name + "_pre")
    for i_block in range(num_blocks):
        tensor = resblock_module(
            tensor, num_filters//2, mid_filters,
            name=f"{name}_block{i_block + 1}")
    tensor = conv2d_bn_mish(
        tensor, mid_filters, 1, name=name + "_post")
    tensor = Concatenate(name=name + "_concat")([tensor, cross_tensor])
    tensor = conv2d_bn_mish(
        tensor, num_filters, 1, name=name + "_out")
    return tensor


def csp_darknet_body(tensor):
    """CSPDarkNet53 model body."""
    tensor = conv2d_bn_mish(tensor, 32, 3, name="conv1")
    tensor = resstage_module(tensor, 64, 1, False, name="stage1")
    tensor = resstage_module(tensor, 128, 2, name="stage2")
    tensor = resstage_module(tensor, 256, 8, name="stage3")
    tensor = resstage_module(tensor, 512, 8, name="stage4")
    tensor = resstage_module(tensor, 1024, 4, name="stage5")
    return tensor


def make_last_layers(tensor, num_filters, name="last1"):
    """5 conv2d_bn_leaky layers followed by a Conv2D layer"""
    tensor = conv2d_bn_leaky(
        tensor, num_filters, 1, name=f"{name}_1")
    tensor = conv2d_bn_leaky(
        tensor, num_filters*2, 3, name=f"{name}_2")
    tensor = conv2d_bn_leaky(
        tensor, num_filters, 1, name=f"{name}_3")
    tensor = conv2d_bn_leaky(
        tensor, num_filters*2, 3, name=f"{name}_4")
    tensor = conv2d_bn_leaky(
        tensor, num_filters, 1, name=f"{name}_5")

    return tensor


def spp_module(tensor, pool_size_list=[(13, 13), (9, 9), (5, 5)],
               name="spp"):
    """Spatial pyramid pooling module."""
    maxpool_tensors = []
    for i_pool, pool_size in enumerate(pool_size_list):
        maxpool_tensors.append(MaxPooling2D(
            pool_size=pool_size, strides=(1, 1),
            padding="same", name=f"{name}_pool{i_pool + 1}")(tensor))
    tensor = Concatenate(name=name + "_concat")([*maxpool_tensors, tensor])
    return tensor


def yolo_keras_app_body(model_func,
        input_shape=(448, 448, 3),
        pretrained_weights="imagenet",
        pan_ids:list=[],
        num_filters=512):
    """Create any model body from
       tensorflow.python.keras.applications."""
    appnet = model_func(
        include_top=False,
        weights=pretrained_weights,
        input_shape=input_shape)
    input_tensor = appnet.input

    i_td = 1
    tensor_i = conv2d_bn_leaky(
        appnet.output, num_filters, 1, name="pan_td1_1")
    tensor_i = conv2d_bn_leaky(
        tensor_i, num_filters*2, 3, name="pan_td1_2")
    tensor_i = conv2d_bn_leaky(
        tensor_i, num_filters, 1, name="pan_td1_spp_pre")
    tensor_i = spp_module(tensor_i, name="pan_td1_spp")
    tensor_i = conv2d_bn_leaky(
        tensor_i, num_filters, 1, name="pan_td1_3")
    tensor_i = conv2d_bn_leaky(
        tensor_i, num_filters*2, 3, name="pan_td1_4")
    tensor_i = conv2d_bn_leaky(
        tensor_i, num_filters, 1, name="pan_td1_5")

    td_output_list = [tensor_i]

    for idx in pan_ids:
        num_filters //= 2
        tensor_i_up = conv2d_bn_leaky(
            tensor_i, num_filters, 1, name=f"pan_td{i_td}_up")
        tensor_i_up = UpSampling2D(2, name=f"pan_td{i_td}_up")(tensor_i_up)
        i_td += 1

        tensor_i = conv2d_bn_leaky(
            appnet.layers[idx].output, num_filters, 1, name=f"pan_td{i_td}_pre")
        tensor_i = Concatenate(name=f"pan_td{i_td}_concat")([tensor_i, tensor_i_up])
        tensor_i = make_last_layers(tensor_i, num_filters, name=f"pan_td{i_td}")
        td_output_list.append(tensor_i)

    output_i = conv2d_bn_leaky(
        tensor_i, num_filters*2, 3, name="pan_out_1")
    output_list = [output_i]

    for i_bu, tensor_td in enumerate(td_output_list[-2::-1]):
        num_filters *= 2
        tensor_i_dn = ZeroPadding2D(
            ((1, 0),(1, 0)), name=f"pan_bu{i_bu + 1}_dn_pad")(tensor_i)
        tensor_i_dn = conv2d_bn_leaky(
            tensor_i_dn, num_filters, 3, strides=(2, 2),
            name=f"pan_bu{i_bu + 1}_dn")
        tensor_i = Concatenate(
            name=f"pan_bu{i_bu + 1}_concat")([tensor_i_dn, tensor_td])
        tensor_i = make_last_layers(tensor_i, num_filters, name=f"pan_bu{i_bu + 1}")

        output_i = conv2d_bn_leaky(
            tensor_i, num_filters*2, 3, name=f"pan_out_{i_bu + 2}")
        output_list.append(output_i)

    return Model(input_tensor, output_list[::-1])
