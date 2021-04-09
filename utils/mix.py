from tensorflow.keras.applications.vgg16 import preprocess_input


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


def last_5chars(x):
    return x[-10:-5]


def vgg_preprocessor(image_sample):
    result = preprocess_input(image_sample)

    return result


def inception_preprocessor(image_sample):
    image_sample /= 255.
    image_sample -= 0.5
    image_sample *= 2.

    return image_sample
