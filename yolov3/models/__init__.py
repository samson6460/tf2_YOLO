"""yolov3.models.__int__"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Multiply
from .backbone import DarknetConv2D
from .backbone import yolo_keras_app_body
from .darknet import darknet53
from .darknet import yolo_body, tiny_yolo_body


def yolo_head(model_body, class_num=10,
              anchors=[[0.89663461, 0.78365384],
                       [0.37500000, 0.47596153],
                       [0.27884615, 0.21634615],
                       [0.14182692, 0.28605769],
                       [0.14903846, 0.10817307],
                       [0.07211538, 0.14663461],
                       [0.07932692, 0.05528846],
                       [0.03846153, 0.07211538],
                       [0.02403846, 0.03125000]]):
    """YOLOv3 head."""
    anchors = np.array(anchors)
    inputs = model_body.input
    output = model_body.output
    tensor_num = len(output)

    if len(anchors)%tensor_num > 0:
        raise ValueError(
            "The total number of anchor boxs "
            "should be a multiple of the number "
            f"{tensor_num} of output tensors")
    abox_num = len(anchors)//tensor_num

    outputs_list = []
    for i_tensor, output_tensor in enumerate(output):
        output_list = []
        start_i = i_tensor*abox_num
        for i_box, box in enumerate(anchors[start_i:start_i + abox_num]):
            xy_output = DarknetConv2D(
                2, 1, activation='sigmoid',
                name=f"out{i_tensor + 1}_box{i_box + 1}_xy_conv")(
                    output_tensor)
            wh_output = DarknetConv2D(
                2, 1, activation='exponential',
                name=f"out{i_tensor + 1}_box{i_box + 1}_wh_conv")(
                    output_tensor)
            wh_output = Multiply(
                name=f"out{i_tensor + 1}_box{i_box + 1}_multiply")(
                    [wh_output, box.reshape((1, 1, 1, 2))])
            c_output = DarknetConv2D(
                1, 1, activation='sigmoid',
                name=f"out{i_tensor + 1}_box{i_box + 1}_conf_conv")(
                    output_tensor)
            p_output = DarknetConv2D(
                class_num, 1, activation='sigmoid',
                name=f"out{i_tensor + 1}_box{i_box + 1}_prob_conv")(
                    output_tensor)
            output_list += [xy_output,
                            wh_output,
                            c_output,
                            p_output]

        outputs = Concatenate(name=f"out{i_tensor + 1}_concat")(output_list)
        outputs_list.append(outputs)

    model = Model(inputs, outputs_list)

    return model
