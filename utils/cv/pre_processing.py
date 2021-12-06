import numpy as np
import torch
import utils.misc as utils


def pre_process(input_array, pre_processing_approach: str, color_model=None):
    """
    A function delegating further pre-processing work to proper function.

    :param input_array: numpy array containing image data
    :param pre_processing_approach: string naming the pre-processing approach to be applied
    :param color_model: color model to be used if pre-processing depends on the order
    :return: numpy array containing pre-processed image data
    """
    if pre_processing_approach == "SSD":
        return pre_process_ssd(input_array)
    if pre_processing_approach == "SSD_2":
        return pre_process_ssd_2(input_array)
    if pre_processing_approach == "YOLO":
        return pre_process_yolo(input_array)
    if pre_processing_approach == "SqueezeNet":
        return pre_process_squeezenet(input_array)
    if pre_processing_approach == "VGG":
        return pre_process_vgg(input_array, color_model)
    if pre_processing_approach == "Inception":
        return pre_process_inception(input_array)
    if pre_processing_approach == "PyTorch":
        return pre_process_py(input_array)
    if pre_processing_approach == "PyTorch_objdet":
        return pre_process_py_objdet(input_array)
    utils.print_goodbye_message_and_die(f"Pre-processing approach \"{pre_processing_approach}\" undefined.")


def pre_process_ssd(input_array):
    """
    A function pre-processing an input array in the way expected by some SSD models.

    Values are converted from int 0 <-> 255 range to float range of -1.0 <-> 1.0.

    :param input_array: numpy array containing image data
    :return: numpy array containing pre-processed image data
    """
    input_array = input_array.astype("float32")

    input_array *= (2.0 / 255.0)
    input_array -= 1.0
    # kinda equivalent solution:
    # input_array -= 127.5
    # input_array *= 0.007843
    return input_array


def pre_process_ssd_2(input_array):
    """
    A function pre-processing an input array in the way expected by some other SSD models.

    Values are converted from int 0 <-> 255 range to float range of -1.0 <-> 1.0.

    :param input_array: numpy array containing image data
    :return: numpy array containing pre-processed image data
    """
    input_array = input_array.astype("float32")
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)

    input_array /= 255.0
    input_array = (input_array - mean) / std
    return input_array


def pre_process_yolo(input_array):
    """
    A function pre-processing an input array in the way expected by some YOLO models.

    Values are converted from int 0 <-> 255 range to float range of 0.0 <-> 1.0. (IN RATHER DIRECT WAY)

    :param input_array: numpy array containing image data
    :return: numpy array containing pre-processed image data
    """
    input_array = input_array.astype("float32")

    input_array /= 255.0
    return input_array


def pre_process_squeezenet(input_array):
    """
    A function pre-processing an input array in the way expected by SqueezeNet model.

    Values are converted from int 0 <-> 255 range to float range of 0.0 <-> 1.0
    and normalized using mean = [0.485, 0.456, 0.406].

    :param input_array: numpy array containing image data
    :return: numpy array containing pre-processed image data
    """
    input_array = input_array.astype("float32")

    input_array /= 255.0

    r_mean = 0.485
    g_mean = 0.456
    b_mean = 0.406

    per_channel_means = np.array([r_mean, g_mean, b_mean])

    input_array -= per_channel_means

    return input_array


def pre_process_vgg(input_array, color_model: str):
    """
    A function pre-processing an input array in the way described in the original VGG paper.

    Values are converted from int 0 <-> 255 range to asymmetrical float ranges different for every color channel.
    Pre-processing is used by various classification models other than VGG, for example ResNet.

    :param input_array: numpy array containing image data
    :param color_model: str, color model of image data, possible values: ["RGB", "BGR"]
    :return: numpy array, array after subtracting the mean RGB values
    """
    if color_model not in ["RGB", "BGR"]:
        utils.print_goodbye_message_and_die(f"Color model {color_model} is not supported.")

    input_array = input_array.astype("float32")

    r_mean = 123.68
    g_mean = 116.779
    b_mean = 103.939

    per_channel_means = np.array([r_mean, g_mean, b_mean])
    if color_model == "BGR":
        per_channel_means = np.flip(per_channel_means)

    input_array -= per_channel_means
    return input_array


def pre_process_inception(input_array):
    """
    A function pre-processing an input array in the way described in the original Inception paper.

    Values are converted from int 0 <-> 255 range to float range of -1.0 <-> 1.0.
    Pre-processing is used by various classification models other than Inception, for example MobileNet.

    :param input_array: numpy array containing image data
    :return: numpy array containing pre-processed image data
    """

    input_array = input_array.astype("float32")

    input_array /= 255.
    input_array -= 0.5
    input_array *= 2.

    return input_array


def pre_process_py(input_array):
    """
    Preprocessing approach for pytorch classification models

    All pre-trained classification models expect input images normalized in the same way, i.e. mini-batches of 3-channel
    RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a
    range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    :param input_array:
    :return:
    """

    per_channel_means = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    input_array /= 255.0
    input_array = (input_array - per_channel_means) / std

    input_array = input_array.astype("float32")

    return input_array


def pre_process_py_objdet(input_array):
    """
    Preprocessing approach for pytorch torchvision object detection models

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each image, and should
    be in 0-1 range. Different images can have different sizes but they will be resized to a fixed size before passing
    it to the backbone
    :param input_array:
    :return:
    """

    # print(input_array)

    preprocessed_input_array = []
    for x in input_array:
        x_casted = x.astype("float32")
        x_casted /= 255.0
        preprocessed_input_array.append(torch.from_numpy(x_casted))

    return preprocessed_input_array
