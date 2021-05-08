from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization as BN


def Conv2D_BN_Leaky(input_tensor, *args):
    output_tensor = Conv2D(*args, 
                           padding='same',
                           kernel_initializer='he_normal')(input_tensor)
    output_tensor = BN()(output_tensor)
    output_tensor = LeakyReLU(alpha=0.1)(output_tensor)
    return output_tensor


def Conv2D_Acti_BN(input_tensor, activation, *args):
    output_tensor = Conv2D(*args,
                           activation=activation,
                           padding='same',
                           kernel_initializer='he_normal')(input_tensor)
    output_tensor = BN()(output_tensor)
    return output_tensor


def UpConv2D_Acti_BN(input_tensor, activation, *args):
    output_tensor = UpSampling2D(size = (2,2))(input_tensor)
    output_tensor = Conv2D(*args,
                           activation=activation,
                           padding='same',
                           kernel_initializer='he_normal')(output_tensor)
    output_tensor = BN()(output_tensor)
    return output_tensor


def darknet_body(input_tensor):
    conv1 = Conv2D_BN_Leaky(input_tensor, 32, 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_BN_Leaky(pool1, 64, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_BN_Leaky(pool2, 128, 3)
    conv3 = Conv2D_BN_Leaky(conv3, 64, 1)
    conv3 = Conv2D_BN_Leaky(conv3, 128, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D_BN_Leaky(pool3, 256, 3)
    conv4 = Conv2D_BN_Leaky(conv4, 128, 1)
    conv4 = Conv2D_BN_Leaky(conv4, 256, 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D_BN_Leaky(pool4, 512, 3)
    conv5 = Conv2D_BN_Leaky(conv5, 256, 1)
    conv5 = Conv2D_BN_Leaky(conv5, 512, 3)
    conv5 = Conv2D_BN_Leaky(conv5, 256, 1)
    conv5 = Conv2D_BN_Leaky(conv5, 512, 3)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D_BN_Leaky(pool5, 1024, 3)
    conv6 = Conv2D_BN_Leaky(conv6, 512, 1)
    conv6 = Conv2D_BN_Leaky(conv6, 1024, 3)
    conv6 = Conv2D_BN_Leaky(conv6, 512, 1)
    output_tensor = Conv2D_BN_Leaky(conv6, 1024, 3)

    return output_tensor


def unet_body(input_tensor):
    conv1 = Conv2D_Acti_BN(input_tensor, "relu", 64, 3)
    conv1 = Conv2D_Acti_BN(conv1, "relu", 64, 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_Acti_BN(pool1, "relu", 128, 3)
    conv2 = Conv2D_Acti_BN(conv2, "relu", 128, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_Acti_BN(pool2, "relu", 256, 3)
    conv3 = Conv2D_Acti_BN(conv3, "relu", 256, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D_Acti_BN(pool3, "relu", 512, 3)
    conv4 = Conv2D_Acti_BN(conv4, "relu", 512, 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D_Acti_BN(pool4, "relu", 1024, 3)
    conv5 = Conv2D_Acti_BN(conv5, "relu", 1024, 3)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    up6 = UpConv2D_Acti_BN(pool5, "relu", 512, 2)
    merge6 = concatenate([conv5, up6], axis = 3)
    conv6 = Conv2D_Acti_BN(merge6, "relu", 512, 3)
    conv6 = Conv2D_Acti_BN(conv6, "relu", 512, 3)

    up7 = UpConv2D_Acti_BN(conv6, "relu", 256, 2)
    merge7 = concatenate([conv4, up7], axis = 3)
    conv7 = Conv2D_Acti_BN(merge7, "relu", 256, 3)
    output_tensor = Conv2D_Acti_BN(conv7, "relu", 256, 3)

    return output_tensor