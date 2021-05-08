import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import MobileNetV2
from .backbone import Conv2D_BN_Leaky
from .backbone import darknet_body
from .backbone import unet_body


def darknet19(input_shape=(416, 416, 3), class_num=10):
    inputs = Input(input_shape)
    darknet_outputs = darknet_body(inputs)

    conv = Conv2D_BN_Leaky(darknet_outputs, class_num, 1)
    x = GlobalAveragePooling2D()(conv)
    outputs = Softmax()(x)

    model = Model(inputs, outputs)

    return model


def yolo_body(input_shape=(416, 416, 3),
              backbone="darknet",
              pretrained_backbone=None):
    inputs = Input(input_shape)
    if backbone == "darknet":
        darknet = Model(inputs, darknet_body(inputs))
        if pretrained_backbone is not None:
            darknet.set_weights(pretrained_backbone.get_weights())

        passthrough = darknet.layers[43].output
        conv = Conv2D_BN_Leaky(darknet.output, 1024, 3)
        conv = Conv2D_BN_Leaky(conv, 1024, 3)

        passthrough = Conv2D_BN_Leaky(passthrough, 64, 3)
        passthrough = tf.nn.space_to_depth(passthrough, 2)

        merge = concatenate([passthrough, conv], axis=-1)

        outputs = Conv2D_BN_Leaky(merge, 1024, 3)
    elif backbone == "unet":
        if pretrained_backbone is None:
            outputs = unet_body(inputs) 
        else:
            outputs = pretrained_backbone(inputs)
    elif backbone == "mobilenet":
        mobilenet = MobileNetV2(input_shape=input_shape,
                                alpha=1.0, include_top=False,
                                weights=pretrained_backbone)
        outputs = mobilenet(inputs)
    else:
        raise ValueError("Invalid backbone: %s" % backbone)
    model = Model(inputs, outputs)
    return model


def yolo_head(model_body, class_num=10, 
              anchors=[(0.04405615, 0.05210654),
                       (0.14418923, 0.15865615),
                       (0.25680231, 0.42110308),
                       (0.60637077, 0.27136769),
                       (0.75157846, 0.70525231)]):
    anchors = np.array(anchors)
    inputs = model_body.input
    output = model_body.output
    output_list = []
    for box in anchors:
        xy_output = Conv2D(2, 1,
                           padding='same',
                           activation='sigmoid',
                           kernel_initializer='he_normal')(output)
        wh_output = Conv2D(2, 1,
                           padding='same',
                           activation='exponential',
                           kernel_initializer='he_normal')(output)
        wh_output = wh_output * box
        c_output = Conv2D(1, 1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal')(output)
        p_output = Conv2D(class_num, 1,
                          padding = 'same',
                          activation='softmax',
                          kernel_initializer='he_normal')(output)
        output_list += [xy_output,
                        wh_output,
                        c_output,
                        p_output]

    outputs = concatenate(output_list, axis=-1)

    model = Model(inputs, outputs)    

    return model