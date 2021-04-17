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
