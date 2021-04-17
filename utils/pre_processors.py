import utils.misc as utils


def pre_process_ssd(input_array):
    """
    A function pre-processing an input array in the way expected by some SSD models.

    Pixel RGB values are converted from int 0 <-> 255 range to float range of -1.0 <-> 1.0.

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


def pre_process_vgg(input_array, color_model: str):
    """
    A function pre-processing an input array in the way described in original VGG paper.

    :param input_array: numpy array containing image data
    :param color_model: str, color model of image data, possible values: ["RGB", "BGR"]
    :return: numpy array, array after subtracting the mean RGB values
    """
    if color_model not in ["RGB", "BGR"]:
        utils.print_goodbye_message_and_die(f"Color model {color_model} is not supported.")

    r_mean = 123.68
    g_mean = 116.779
    b_mean = 103.939

    per_channel_means = np.array([r_mean, g_mean, b_mean])
    if color_model == "BGR":
        per_channel_means = np.flip(per_channel_means)

    input_array -= per_channel_means
    return input_array
