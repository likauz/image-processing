from scipy.signal import convolve2d
import numpy as np
import imageio as iio
import skimage
import scipy
import scipy.signal as sps
from skimage.color import rgb2gray


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    frames = iio.imread(filename).astype('float64')
    if representation == 1:
        rgb2gray = skimage.color.rgb2gray(frames)
        return np.divide(rgb2gray, 255.0)
    else:
        return np.divide(frames, 255.0)


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    # BLUR
    blur_image = scipy.ndimage.filters.convolve(im, blur_filter)
    blur_image = scipy.ndimage.filters.convolve(blur_image, blur_filter.T)

    # REDUCE
    reduce_image = blur_image[0::2, 0::2]
    return reduce_image


def calc_filter_vec(filter_size):
    """
       This function calculate the filter vector that appropriate to the given filter size
       :param filter_size: the size of the vector
       :return: filter vector with filter_size size
       """
    conv_i = np.reshape([1, 1], (1, 2))
    a = np.reshape([1, 1], (1, 2))
    k = (filter_size-1)/2
    conv_i = sps.convolve2d(conv_i, conv_i)
    for i in range(1, filter_size-2):
        conv_i = sps.convolve2d(conv_i, a)
    return conv_i / np.power(2, 2*k)


def build_gaussian_pyramid(im, max_levels, filter_size):  # ok
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """

    # find the filter_vec
    filter_vec = calc_filter_vec(filter_size)

    # find the Gaussian Pyramids
    pyr = []
    i_level_image = im
    pyr.append(im)
    for i in range(1, max_levels):
        height = i_level_image.shape[0]/2
        width = i_level_image.shape[1]/2
        if height >= 16 and width >= 16:
            i_level_image = reduce(i_level_image, filter_vec)
            pyr.append(i_level_image)
        else:
            break
    return pyr, filter_vec

def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img



