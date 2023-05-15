"""Darknet definition for YOLOv4."""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import get_file
from .backbone import conv2d_bn_leaky, csp_darknet_body
from .backbone import make_last_layers, spp_module


WEIGHTS_PATH_YOLOV4_BODY = "https://github.com/samson6460/tf2_YOLO/releases/download/YOLOv4/tf_keras_yolov4_608_body.h5"
WEIGHTS_PATH_YOLOV4_MODEL = "https://github.com/samson6460/tf2_YOLO/releases/download/YOLOv4/tf_keras_yolov4_608_model.h5"
WEIGHTS_PATH_CSPDN53_TOP = "https://github.com/samson6460/tf2_YOLO/releases/download/YOLOv4/tf_keras_darknet53_448_include_top.h5"
WEIGHTS_PATH_CSPDN53_NOTOP = "https://github.com/samson6460/tf2_YOLO/releases/download/YOLOv4/tf_keras_darknet53_448_no_top.h5"


def csp_darknet53(include_top=True, weights="imagenet",
                  input_shape=(448, 448, 3), class_num=1000):
    """Create Darknet53 model.

    Args:
        include_top: A boolean, whether to include the fully-connected layer
            at the top of the network.
        weights: one of None (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_shape: A tuple of 3 integers,
            shape of input image.
        class_num: optional number of classes to classify images into,
            only to be specified if include_top is True,
            and if no weights argument is specified.
    """
    inputs = Input(input_shape)
    outputs = csp_darknet_body(inputs)

    if include_top:
        tensor = GlobalAveragePooling2D()(outputs)
        outputs = Dense(class_num, activation="softmax")(tensor)

    model = Model(inputs, outputs)

    if weights is not None:
        if weights == "imagenet":
            if include_top:
                if ((input_shape[0]%32 > 0) or
                    (input_shape[1]%32 > 0) or
                    (input_shape[2] != 3)):
                    raise ValueError("When setting `include_top=True` "
                        "and loading `imagenet` weights, "
                        "`input_shape` should be (32x, 32x, 3).")
                if class_num != 1000:
                    raise ValueError("If using `weights` as `'imagenet'` "
                        "with `include_top` as true, "
                        "`class_num` should be 1000")
                weights = get_file(
                    "tf_keras_cspdarknet53_448_include_top.h5",
                    WEIGHTS_PATH_CSPDN53_TOP,
                    cache_subdir="models")
            else:
                weights = get_file(
                    "tf_keras_cspdarknet53_448_no_top.h5",
                    WEIGHTS_PATH_CSPDN53_NOTOP,
                    cache_subdir="models")
        model.load_weights(weights)

    return model


def yolo_body(input_shape=(608, 608, 3),
              pretrained_darknet=None,
              pretrained_weights=None):
    """Create YOLOv4 body in tf.keras."""
    input_tensor = Input(input_shape)
    darknet = Model(input_tensor, csp_darknet_body(input_tensor))
    if pretrained_darknet is not None:
        darknet.set_weights(pretrained_darknet.get_weights())

    tensor_s = conv2d_bn_leaky(
        darknet.output, 512, 1, name="pan_td1_1")
    tensor_s = conv2d_bn_leaky(
        tensor_s, 1024, 3, name="pan_td1_2")
    tensor_s = conv2d_bn_leaky(
        tensor_s, 512, 1, name="pan_td1_spp_pre")
    tensor_s = spp_module(tensor_s, name="pan_td1_spp")
    tensor_s = conv2d_bn_leaky(
        tensor_s, 512, 1, name="pan_td1_3")
    tensor_s = conv2d_bn_leaky(
        tensor_s, 1024, 3, name="pan_td1_4")
    tensor_s = conv2d_bn_leaky(
        tensor_s, 512, 1, name="pan_td1_5")

    tensor_s_up = conv2d_bn_leaky(
        tensor_s, 256, 1, name="pan_td1_up")
    tensor_s_up = UpSampling2D(2, name="pan_td1_up")(tensor_s_up)

    tensor_m = conv2d_bn_leaky(
        darknet.layers[204].output, 256, 1, name="pan_td2_pre")
    tensor_m = Concatenate(name="pan_td1_concat")([tensor_m, tensor_s_up])
    tensor_m = make_last_layers(tensor_m, 256, name="pan_td2")

    tensor_m_up = conv2d_bn_leaky(
        tensor_m, 128, 1, name="pan_td2_up")
    tensor_m_up = UpSampling2D(2, name="pan_td2_up")(tensor_m_up)

    tensor_l = conv2d_bn_leaky(
        darknet.layers[131].output, 128, 1, name="pan_td3_pre")
    tensor_l = Concatenate(name="pan_td2_concat")([tensor_l, tensor_m_up])
    tensor_l = make_last_layers(tensor_l, 128, name="pan_td3")

    output_l = conv2d_bn_leaky(
        tensor_l, 256, 3, name="pan_out_l")

    tensor_l_dn = ZeroPadding2D(
        ((1, 0),(1, 0)), name="pan_bu1_dn_pad")(tensor_l)
    tensor_l_dn = conv2d_bn_leaky(
        tensor_l_dn, 256, 3, strides=(2, 2), name="pan_bu1_dn")
    tensor_m = Concatenate(name="pan_bu1_concat")([tensor_l_dn, tensor_m])
    tensor_m = make_last_layers(tensor_m, 256, name="pan_bu1")

    output_m = conv2d_bn_leaky(
        tensor_m, 512, 3, name="pan_out_m")

    tensor_m_dn = ZeroPadding2D(
        ((1, 0),(1, 0)), name="pan_bu2_dn_pad")(tensor_m)
    tensor_m_dn = conv2d_bn_leaky(
        tensor_m_dn, 512, 3, strides=(2, 2), name="pan_bu2_dn")
    tensor_s = Concatenate(name="pan_bu2_concat")([tensor_m_dn, tensor_s])
    tensor_s = make_last_layers(tensor_s, 512, name="pan_bu2")

    output_s = conv2d_bn_leaky(
        tensor_s, 1024, 3, name="pan_out_s")

    model = Model(input_tensor, [output_s, output_m, output_l])

    if pretrained_weights is not None:
        if pretrained_weights == "ms_coco":
            pretrained_weights = get_file(
                "tf_keras_yolov4_body.h5",
                WEIGHTS_PATH_YOLOV4_BODY,
                cache_subdir="models")
        model.load_weights(pretrained_weights)

    return model
