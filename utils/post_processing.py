import cv2
import numpy as np
from utils.labels import imagenet_labels


def initialize_colors(colors_to_generate=100):
    """
    A function randomly assigning colors to category id.

    :param colors_to_generate: int, number of colors to generate
    :return: list of tuples equating to BGR-expressed colors
    """
    colors_per_cat = list()
    np.random.seed(13)  # 13
    for _ in range(colors_to_generate):
        B = np.random.randint(0, 256)
        G = np.random.randint(0, 256)
        R = np.random.randint(0, 256)
        colors_per_cat.append((B, G, R))
    return colors_per_cat


COLORS_PER_CAT = initialize_colors()


def draw_line(image, cat, x_0, y_0, x_1, y_1):
    """
    A function drawing a line with cv2 facilities.

    :param image: image to put line onto
    :param cat: category id line refers to
    :param x_0: first horizontal coordinate
    :param y_0: first vertical coordinate
    :param x_1: second horizontal coordinate
    :param y_1: second vertical coordinate
    :return: image with line applied
    """
    image = cv2.line(image,
                     (x_0, y_0),
                     (x_1, y_1),
                     color=COLORS_PER_CAT[cat],
                     lineType=cv2.LINE_AA,
                     thickness=2)
    return image


def draw_bbox(image, bbox, cat):
    """
    A function creating a bbox from four lines.

    :param image: image to apply bbox onto
    :param bbox: list containing bbox boundaries in absolute values
    :param cat: category that bbox refers to
    :return: image with bbox applied
    """
    bbox = [int(elem) for elem in bbox]
    image = draw_line(image, cat, bbox[0], bbox[1], bbox[2], bbox[1])
    image = draw_line(image, cat, bbox[0], bbox[1], bbox[0], bbox[3])
    image = draw_line(image, cat, bbox[0], bbox[3], bbox[2], bbox[3])
    image = draw_line(image, cat, bbox[2], bbox[3], bbox[2], bbox[1])
    return image


def get_imagenet_names(ids_array):
    """
    A function translating numpy array with ImageNet category ids to their readable counterparts.

    :param ids_array: numpy array with ImageNet category ids
    :return: list of strings with ImageNet category names
    """
    list_of_names = list()
    for id in ids_array:
        list_of_names.append(imagenet_labels[id])
    return list_of_names
