# Converts a grayscale image into a numpy array of values normalized to 0-1.
# Also can convert a list of classes into a one hot encoded version of it in an np array.

from PIL import Image
import numpy as np

def to_mnist(image_path, invert=False):
    assert type(image_path) == str, "'image_path' must be a string"
    image = Image.open(image_path)
    pixels = image.load()
    assert type(pixels[0, 0]) == int, "Image must be grayscale"
    values = np.zeros(image.size)
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            if invert:
                values[i, j] = 1 - pixels[i, j] / 255
            else:
                values[i, j] = pixels[i, j] / 255
    return values

def convert_mass(header, file_type, names, dim=(28, 28), invert=False):
    """
    header: Images/ \n
    file_type: .png \n
    names: ['img1', 'img2', ..., 'imgN'] \n
    MNIST type version of Images/img1.png, Images/img2.png, ..., Images/imgN.png
    """
    images = np.zeros((len(names), dim[0], dim[1], 1))
    for x, name in enumerate(names):
        image_path = header+name+file_type
        image = Image.open(image_path)
        pixels = image.load()
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                if invert:
                    images[x, i, j, 0] = 1 - pixels[i, j] / 255
                else:
                    images[x, i, j, 0] = pixels[i, j] / 255
    return images

def one_hot_classes(all_classes, types):
    """
    all_classes: ['bat', 'fish', 'cat'] \n
    types: ['cat', 'cat', 'fish', 'bat', 'fish'] \n
    Numpy array of [
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0]
    ]
    """
    one_hot = np.zeros((len(types), len(all_classes)))
    for i, class_type in enumerate(types):
        one_hot[i, all_classes.index(class_type)] = 1
    return one_hot
