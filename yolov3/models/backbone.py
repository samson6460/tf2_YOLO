"""YOLO_v3 Model Defined in Keras.

Modified from https://github.com/qqwweee/keras-yolo3."""

from functools import wraps
from functools import reduce

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Add
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_initializer': 'he_normal'}
    if kwargs.get('strides') == (2, 2):
        darknet_conv_kwargs['padding'] = 'valid'
    else:
        darknet_conv_kwargs['padding'] = 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for _ in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters//2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def resblock_module(x, num_filters, num_blocks):
    y = DarknetConv2D_BN_Leaky(num_filters, (1, 1), 2)(x)
    x = compose(DarknetConv2D_BN_Leaky(num_filters//2, (1, 1), 2),
                DarknetConv2D_BN_Leaky(num_filters//2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    x = Add()([x, y])
    for _ in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters//2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters//2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def resnet_body(x):
    '''Resnent body having 90 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_module(x, 64, 1)
    x = resblock_module(x, 128, 2)
    x = resblock_module(x, 256, 8)
    x = resblock_module(x, 512, 8)
    x = resblock_module(x, 1024, 4)
    return x


def make_last_layers(x, num_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3, 3))(x)
    return x, y


def yolo_keras_app_body(model_func,
        input_shape=(416, 416, 3),
        pretrained_weights="imagenet",
        fpn_id=[],
        num_filters=512):
    """Create any model body from
       tensorflow.python.keras.applications."""
    appnet = model_func(
        include_top=False,
        weights=pretrained_weights,
        input_shape=input_shape)
    inputs = appnet.input
    
    x, y = make_last_layers(appnet.output, num_filters)
    output_list = [y]

    for id in fpn_id:
        num_filters //= 2
        x = compose(
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                UpSampling2D(2))(x)
        x = Concatenate()([x, appnet.layers[id].output])
        x, y = make_last_layers(x, num_filters)
        output_list.append(y)

    return Model(inputs, output_list)