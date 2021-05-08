from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from .backbone import DarknetConv2D_BN_Leaky
from .backbone import darknet_body, make_last_layers
from .backbone import compose


def darknet53(input_shape=(416, 416, 3), class_num=10):
    inputs = Input(input_shape)
    darknet_outputs = darknet_body(inputs)

    x = GlobalAveragePooling2D()(darknet_outputs)
    outputs = Dense(class_num, activation="softmax")(x)

    model = Model(inputs, outputs)

    return model


def yolo_body(input_shape=(416, 416, 3),
              pretrained_darknet=None,
              pretrained_weights=None):
    """Create YOLO_V3 model CNN body in Keras."""
    inputs = Input(input_shape)
    darknet = Model(inputs, darknet_body(inputs))
    if pretrained_darknet is not None:
        darknet.set_weights(pretrained_darknet.get_weights())

    x, y1 = make_last_layers(darknet.output, 512)

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256)

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128)
    model = Model(inputs, [y1, y2, y3])

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model


def tiny_yolo_body(input_shape=(416, 416, 3)):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    inputs = Input(input_shape)
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = DarknetConv2D_BN_Leaky(512, (3,3))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)))([x2, x1])

    return Model(inputs, [y1, y2])