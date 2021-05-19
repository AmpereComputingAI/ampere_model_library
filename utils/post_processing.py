import numpy as np
import utils.misc as utils


def initialize_colors(colors_to_generate=100):
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
    image = cv2.line(image,
                     (x_0, y_0),
                     (x_1, y_1),
                     color=COLORS_PER_CAT[cat],
                     lineType=cv2.LINE_AA,
                     thickness=2)
    return image


def draw_bbox(image, bbox, cat):
    bbox = [int(elem) for elem in bbox]
    image = draw_line(image, cat, bbox[0], bbox[1], bbox[2], bbox[1])
    image = draw_line(image, cat, bbox[0], bbox[1], bbox[0], bbox[3])
    image = draw_line(image, cat, bbox[0], bbox[3], bbox[2], bbox[3])
    image = draw_line(image, cat, bbox[2], bbox[3], bbox[2], bbox[1])
    return image
