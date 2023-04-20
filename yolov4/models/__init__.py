"""yolov4.models.__int__"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate

from .backbone import DarknetConv2D
from .backbone import Anchor
from .backbone import yolo_keras_app_body
from .darknet import csp_darknet53
from .darknet import yolo_body


def yolo_head(model_body, class_num=80,
              anchors=[[0.75493421, 0.65953947],
                       [0.31578947, 0.39967105],
                       [0.23355263, 0.18092105],
                       [0.11842105, 0.24013158],
                       [0.12500000, 0.09046053],
                       [0.05921053, 0.12335526],
                       [0.06578947, 0.04605263],
                       [0.03125000, 0.05921053],
                       [0.01973684, 0.02631579]]):
    """YOLOv4 head."""
    anchors = np.array(anchors)
    input_tensor = model_body.input
    out_tensors = model_body.output
    tensor_num = len(out_tensors)

    if len(anchors)%tensor_num > 0:
        raise ValueError(
            "The total number of anchor boxs "
            "should be a multiple of the number "
            f"{tensor_num} of output tensors")
    abox_num = len(anchors)//tensor_num

    outputs_list = []
    for i_tensor, out_tensor in enumerate(out_tensors):
        output_list = []
        start_i = i_tensor*abox_num
        for i_box, box in enumerate(anchors[start_i:start_i + abox_num]):
            xy_output = DarknetConv2D(
                2, 1, activation='sigmoid',
                name=f"out{i_tensor + 1}_box{i_box + 1}_xy_conv")(
                    out_tensor)
            wh_output = DarknetConv2D(
                2, 1,
                name=f"out{i_tensor + 1}_box{i_box + 1}_wh_conv")(
                    out_tensor)
            wh_output = Anchor(
                box, name=f"out{i_tensor + 1}_box{i_box + 1}_anchor")(
                    wh_output)
            c_output = DarknetConv2D(
                1, 1, activation='sigmoid',
                name=f"out{i_tensor + 1}_box{i_box + 1}_conf_conv")(
                    out_tensor)
            p_output = DarknetConv2D(
                class_num, 1, activation='sigmoid',
                name=f"out{i_tensor + 1}_box{i_box + 1}_prob_conv")(
                    out_tensor)
            output_list += [xy_output,
                            wh_output,
                            c_output,
                            p_output]

        outputs = Concatenate(name=f"out{i_tensor + 1}_concat")(output_list)
        outputs_list.append(outputs)

    model = Model(input_tensor, outputs_list)

    return model
