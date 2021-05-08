from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D
from .backbone import DarknetConv2D_BN_Leaky
from .backbone import resnet_body, make_last_layers
from .backbone import compose


def resnet90(input_shape=(416, 416, 3), class_num=10):
    inputs = Input(input_shape)
    resnet_outputs = resnet_body(inputs)

    x = GlobalAveragePooling2D()(resnet_outputs)
    outputs = Dense(class_num, activation="softmax")(x)

    model = Model(inputs, outputs)

    return model


def yolo_resnet90_body(input_shape=(416, 416, 3),
                       pretrained_resnet=None):
    """Create ResNet90 model body in Keras."""
    inputs = Input(input_shape)
    resnet = Model(inputs, resnet_body(inputs))
    if pretrained_resnet is not None:
        resnet.set_weights(pretrained_resnet.get_weights())

    x, y1 = make_last_layers(resnet.output, 512)

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1, 1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, resnet.layers[245].output])
    x, y2 = make_last_layers(x, 256)

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1, 1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, resnet.layers[152].output])
    x, y3 = make_last_layers(x, 128)

    return Model(inputs, [y1, y2, y3])