
import cv2
import numpy as np

def one_hot_it(label,label_values):
    semantic_map = []
    for colour in label_values:

        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)


    return semantic_map
def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x
def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x