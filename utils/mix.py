import os


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


def calculate_images():
    path_to_images = os.environ['IMAGES_PATH']
    number_of_images = len(os.listdir(path_to_images))
    return number_of_images
