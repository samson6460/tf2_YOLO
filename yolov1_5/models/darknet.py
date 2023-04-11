"""Darknet definition for YOLOv1."""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from .backbone import darknet_body


def darknet(input_shape=(224, 224, 3), class_num=10):
    """DarkNetv1 model."""
    inputs = Input(input_shape)
    darknet_outputs = darknet_body(inputs)

    tensor = GlobalAveragePooling2D()(darknet_outputs)
    outputs = Dense(class_num, activation="softmax")(tensor)

    model = Model(inputs, outputs)

    return model


def yolo_body(input_shape=(448, 448, 3), pretrained_darknet=None):
    """Body of YOLOv1."""
    inputs = Input(input_shape)
    darknet_model = Model(inputs, darknet_body(inputs))

    if pretrained_darknet is not None:
        darknet_model.set_weights(pretrained_darknet.get_weights())

    return darknet_model


def yolo_head(model_body, bbox_num=2, class_num=10):
    """Head of YOLOv1."""
    inputs = model_body.input
    output = model_body.output

    xywhc_output = Conv2D(5*bbox_num, 1,
                          padding='same',
                          kernel_initializer='he_normal',
                          activation='sigmoid')(output)
    p_output = Conv2D(class_num, 1,
                      padding = 'same',
                      kernel_initializer='he_normal',
                      activation='softmax')(output)

    outputs = concatenate([xywhc_output, p_output], axis=3)

    model = Model(inputs, outputs)

    return model
