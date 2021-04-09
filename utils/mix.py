from tensorflow.keras.applications.vgg16 import preprocess_input


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


def last_5chars(x):
    """
    A function which returns 5 last digits in the name of an image in the validation dataset.
    Function is used as a key for sorting the directory with images.
    """
    return x[-10:-5]


def vgg_preprocessor(image_sample):
    """
    A function which returns a preprocessed image.
    :param image_sample: numpy array, an image which will be added to a batch, eg. shape of array (1, 224, 224, 3)
    :return: numpy array, array after subtracting the mean RGB value, computed on the training set, from each pixel
    """
    result = preprocess_input(image_sample)
    print(result)

    return result


def inception_preprocessor(image_sample):
    """
    A function which returns a preprocessed image.
    :param image_sample: numpy array, an image which will be added to a batch, eg. shape of array (1, 224, 224, 3)
    :return: numpy array
    """
    image_sample /= 255.
    image_sample -= 0.5
    image_sample *= 2.
    print (image_sample)

    return image_sample
