import numpy as np
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
    if pre_processing_approach == "YOLO":
        return pre_process_yolo(input_array)
    if pre_processing_approach == "VGG":
        return pre_process_vgg(input_array, color_model)
    if pre_processing_approach == "Inception":
        return pre_process_inception(input_array)
    if pre_processing_approach == "uint8":
        return pre_process_uint8(input_array)
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


def pre_process_uint8(input_array):
    """
    Function casting the input array to uint8 type

    :param input_array: numpy array containing image data
    :return: numpy array containing pre-processed image data
    """

    input_array = input_array.astype("uint8")

    return input_array
