"""Darknet definition for YOLOv3."""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import get_file
from .backbone import DarknetConv2D_BN_Leaky
from .backbone import darknet_body, make_last_layers
from .backbone import compose


WEIGHTS_PATH_DN_BODY = "https://github.com/samson6460/tf2_YOLO/releases/download/1.0/tf_keras_yolov3_body.h5"
WEIGHTS_PATH_DN53_TOP = "https://github.com/samson6460/tf2_YOLO/releases/download/Weights/tf_keras_darknet53_448_include_top.h5"
WEIGHTS_PATH_DN53_NOTOP = "https://github.com/samson6460/tf2_YOLO/releases/download/Weights/tf_keras_darknet53_448_no_top.h5"

def darknet53(include_top=True, weights='imagenet',
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
    outputs = darknet_body(inputs)

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
                    "tf_keras_darknet53_448_include_top.h5",
                    WEIGHTS_PATH_DN53_TOP,
                    cache_subdir="models")
            else:
                weights = get_file(
                    "tf_keras_darknet53_448_no_top.h5",
                    WEIGHTS_PATH_DN53_NOTOP,
                    cache_subdir="models")
        model.load_weights(weights)

    return model


def yolo_body(input_shape=(416, 416, 3),
              pretrained_darknet=None,
              pretrained_weights=None):
    """Create YOLO_V3 model CNN body in Keras."""
    inputs = Input(input_shape)
    darknet = Model(inputs, darknet_body(inputs))
    if pretrained_darknet is not None:
        darknet.set_weights(pretrained_darknet.get_weights())

    tensor, out_tensor1 = make_last_layers(darknet.output, 512)

    tensor = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(tensor)
    tensor = Concatenate()([tensor, darknet.layers[152].output])
    tensor, out_tensor2 = make_last_layers(tensor, 256)

    tensor = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(tensor)
    tensor = Concatenate()([tensor, darknet.layers[92].output])
    tensor, out_tensor3 = make_last_layers(tensor, 128)
    model = Model(inputs, [out_tensor1, out_tensor2, out_tensor3])

    if pretrained_weights is not None:
        if pretrained_weights == "pascal_voc":
            pretrained_weights = get_file(
                "tf_keras_yolov3_body.h5",
                WEIGHTS_PATH_DN_BODY,
                cache_subdir="models")
        model.load_weights(pretrained_weights)

    return model


def tiny_yolo_body(input_shape=(416, 416, 3)):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    inputs = Input(input_shape)
    tensor1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(2, strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(2, strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(2, strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(2, strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)
    tensor2 = compose(
        MaxPooling2D(2, strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(2, strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(tensor1)
    out_tensor1 = DarknetConv2D_BN_Leaky(512, (3, 3))(tensor2)

    tensor2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(tensor2)
    out_tensor2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)))([tensor2, tensor1])

    return Model(inputs, [out_tensor1, out_tensor2])
