import numpy as np

import skimage.transform
from lasagne.utils import floatX
from scipy.misc import imsave

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
plt.interactive(False)


def save_vgg_to_image_file(image_path, x):
    imsave(image_path, vgg_input_to_image(x))


mean_channel_values = np.array([103.939, 116.779, 123.68]).reshape((3, 1, 1))

def compute_shape(img_shape, max_size):
    # initial shape is (h, w, c)
    h, w, c = img_shape
    if w < h:
        # new size param is (h, w), so keep h, scale down w
        return (max_size, int(float(w) / h * max_size), c)
    else:
        # new size param is (h, w), so keep w, scale down h
        return (int(float(h) / w * max_size), max_size, c)


def resize_image_to_max_size(img, max_size):
    h, w, _ = compute_shape(img.shape, max_size)
    return skimage.transform.resize(img, (h, w), preserve_range=True)


def resize_image_to_vgg_input(img, max_size):
    if max_size is not None:
        img = resize_image_to_max_size(img, max_size)

    return image_to_vgg_input(img)


def image_to_vgg_input(img):
    # shuffle axes from hwc to c01
    img = img.transpose((2, 0, 1))
    # convert RGB to BGR
    img = img[::-1, :, :]
    # normalize the values for the VGG net
    img = img - mean_channel_values
    # add a batch dimension for the VGG net
    img = img.reshape([1] + list(img.shape))
    return floatX(img)


def vgg_input_to_image(x):
    # x in bc01 layout
    # get the first element out of the batch dimension
    x = np.copy(x[0])

    # adjust the mean to image range [0-255]
    x += mean_channel_values

    # convert BGR to RGB
    x = x[::-1]

    # shuffle axes from c01 to 01c
    x = x.transpose((1, 2, 0))

    # ensure limits and type of an image
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# make the match dimensions match the desired shape by cropping and padding w/reflection
def image_to_shape(image, desired_shape):
    if image.shape != desired_shape:
        # crop if a dimension is larger
        if image.shape[2] > desired_shape[2] or image.shape[3] > desired_shape[3]:
            # TODO: get a crop that has the most "stuff"
            # one idea is to compare 9 crops (4 corners, 4 center edges, full center) by total variation and choose the one with largest t.v.
            #image = image[:, :, :desired_shape[2], :desired_shape[3]]
            h = image.shape[2]
            w = image.shape[3]
            image = image[:, :, h // 2 - desired_shape[2] // 2:h // 2 + desired_shape[2] // 2, w // 2 - desired_shape[3] // 2:w // 2 + desired_shape[3] // 2]
        # pad if a dimension is smaller
        if image.shape[2] < desired_shape[2] or image.shape[3] < desired_shape[3]:
            r_diff = desired_shape[2] - image.shape[2]
            c_diff = desired_shape[3] - image.shape[3]
            image = np.pad(image, ((0, 0), (0, 0), (r_diff // 2, r_diff - (r_diff // 2)), (c_diff // 2, c_diff - (c_diff // 2))), mode='reflect')
    return image

def show_vgg_image(image):
    plt.figure(figsize=(image.shape[3] / 80., image.shape[2] / 80.))
    plt.axis('off')
    plt.figimage(vgg_input_to_image(image))
    plt.show(block=True)


def remove_reflect_padding(image):
    return image[32:image.shape[0]-32, 32:image.shape[1]-32, :]

def add_reflect_padding(image):
    return np.pad(image, ((32, 32), (32, 32), (0, 0)), mode='reflect')

def remove_reflect_padding_vgg(image_tensor):
    # bc01
    s = image_tensor.shape
    return image_tensor[:, :, 32:s[2]-32, 32:s[3]-32]

