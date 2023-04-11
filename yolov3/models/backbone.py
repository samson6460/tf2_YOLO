"""YOLOv3 Model Defined in Keras.

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


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
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


@wraps(DarknetConv2D)
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    bn_name = None
    acti_name = None
    if "name" in kwargs:
        name = kwargs["name"]
        kwargs["name"] = name + "_conv"
        bn_name = name + "_bn"
        acti_name = name + "_leaky"

    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(name=bn_name),
        LeakyReLU(alpha=0.1, name=acti_name))


def resblock_body(tensor, num_filters, num_blocks, name="block1"):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    tensor = ZeroPadding2D(((1, 0), (1, 0)), name=name + "_pad")(tensor)
    tensor = DarknetConv2D_BN_Leaky(
        num_filters, 3, strides=(2, 2), name=name + "_dn")(tensor)
    for i_block in range(num_blocks):
        main_tensor = compose(
            DarknetConv2D_BN_Leaky(
                num_filters//2, 1, name=f"{name}_{i_block + 1}_1x1"),
            DarknetConv2D_BN_Leaky(
                num_filters, 3, name=f"{name}_{i_block + 1}_3x3"))(tensor)
        tensor = Add(name=f"{name}_{i_block + 1}_add")([tensor, main_tensor])
    return tensor


def darknet_body(tensor):
    '''Darknent body having 52 Convolution2D layers'''
    tensor = DarknetConv2D_BN_Leaky(32, 3, name="conv1")(tensor)
    tensor = resblock_body(tensor, 64, 1, name="block1")
    tensor = resblock_body(tensor, 128, 2, name="block2")
    tensor = resblock_body(tensor, 256, 8, name="block3")
    tensor = resblock_body(tensor, 512, 8, name="block4")
    tensor = resblock_body(tensor, 1024, 4, name="block5")
    return tensor


def make_last_layers(tensor, num_filters, name="last1"):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    tensor = compose(
        DarknetConv2D_BN_Leaky(num_filters, 1, name=f"{name}_1_1x1"),
        DarknetConv2D_BN_Leaky(num_filters*2, 3, name=f"{name}_1_3x3"),
        DarknetConv2D_BN_Leaky(num_filters, 1, name=f"{name}_2_1x1"),
        DarknetConv2D_BN_Leaky(num_filters*2, 3, name=f"{name}_2_3x3"),
        DarknetConv2D_BN_Leaky(num_filters, 1, name=f"{name}_3_1x1"))(tensor)
    out_tensor = DarknetConv2D_BN_Leaky(
        num_filters*2, 3, name=f"{name}_3_3x3")(tensor)
    return tensor, out_tensor


def yolo_keras_app_body(model_func,
        input_shape=(416, 416, 3),
        pretrained_weights="imagenet",
        fpn_id:list=[],
        num_filters=512):
    """Create any model body from
       tensorflow.python.keras.applications."""
    appnet = model_func(
        include_top=False,
        weights=pretrained_weights,
        input_shape=input_shape)
    inputs = appnet.input

    tensor, out_tensor = make_last_layers(
        appnet.output, num_filters, name="last1")
    output_list = [out_tensor]

    for i_fpn, idx in enumerate(fpn_id):
        num_filters //= 2
        tensor = compose(
            DarknetConv2D_BN_Leaky(
                num_filters, 1, name=f"up{i_fpn + 1}"),
            UpSampling2D(2, name=f"up{i_fpn + 1}_up"))(tensor)
        tensor = Concatenate(name=f"concat{i_fpn + 1}")([tensor, appnet.layers[idx].output])
        tensor, out_tensor = make_last_layers(
            tensor, num_filters, name=f"last{i_fpn + 2}")
        output_list.append(out_tensor)

    return Model(inputs, output_list)
