import sys


def batch(iterable, n=1):
    """
    A generator function which yields batches of images.

    :param iterable: iterable, a sorted list of images in the parent directory
    :param n: integer, a batch size
    :return: yields an iterable of batch images
    """
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


def vgg_preprocessor(image):
    """
    A function which returns a preprocessed image.

    :param image: numpy array, an image which will be added to a batch, eg. shape of array (1, 224, 224, 3)
    :param channels: str, order of colors which model accepts
    :return: numpy array, array after subtracting the mean RGB value, computed on the training set, from each pixel
    """

    _R_MEAN = 123.68
    _G_MEAN = 116.779
    _B_MEAN = 103.939

    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    image -= means

    return image


def inception_preprocessor(image_sample):
    """
    A function which returns a preprocessed image.

    :param image_sample: numpy array, numpy array with image to be pre-processed
    :return: numpy array
    """
    image_sample /= 255.
    image_sample -= 0.5
    image_sample *= 2.

    return image_sample


def print_goodbye_message_and_die(message):
    """
    A function printing fail message and making program quit with exit code 1.

    :param message: str
    """
    print(f"\nFAIL: {message}")
    sys.exit(1)


def print_warning_message(message):
    """
    A function printing a warning message but not killing the program.

    :param message: str
    """
    print(f"\nCAUTION: {message}")
