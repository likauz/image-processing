import numpy as np
import scipy
import scipy.signal as sps
import skimage
from imageio import imread
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import os


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


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    # EXPAND
    N, M = im.shape
    expand_image = np.zeros((N*2, M*2), dtype=im.dtype)
    expand_image[::2, ::2] = im

    # BLUR
    blur_image = scipy.ndimage.filters.convolve(expand_image, blur_filter)
    expand_image = scipy.ndimage.filters.convolve(blur_image, blur_filter.T)
    return expand_image


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


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
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

    #  find the Laplacian Pyramids
    gaussian_pyramids, filter = build_gaussian_pyramid(im, max_levels, filter_size)
    levels = len(gaussian_pyramids)
    i = 0
    pyr = []
    while i < levels-1:
        gaussian = gaussian_pyramids[i]
        expand_gaussian = expand(gaussian_pyramids[i+1], 2 * filter_vec)
        i_level_image = gaussian - expand_gaussian
        pyr.append(i_level_image)
        i = i+1
    pyr.append(gaussian_pyramids[levels-1])
    return pyr, filter_vec



def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """

    level = len(lpyr) - 1
    sum_lpyr = lpyr[level] * coeff[level]
    while level != 0:
        sum_lpyr = expand(sum_lpyr, 2 * filter_vec)
        sum_lpyr += lpyr[level-1] * coeff[level-1]
        level = level - 1
    return sum_lpyr


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    N, M = pyr[0].shape
    pyr[0] = np.subtract(pyr[0], np.min(pyr[0])) / np.subtract(np.max(pyr[0]), np.min(pyr[0]))
    res = pyr[0]

    for i in range(1, levels):
        zero_to_add_row = N - pyr[i].shape[0]
        zero_to_add_col = pyr[i].shape[1]
        pyr[i] = np.subtract(pyr[i], np.min(pyr[i])) / np.subtract(np.max(pyr[i]), np.min(pyr[i]))
        zero_to_add_img = np.zeros((zero_to_add_row, zero_to_add_col))
        concat_zero_img = np.concatenate((pyr[i], zero_to_add_img), axis=0)
        res = np.concatenate((res, concat_zero_img), axis=1)
    return res


def display_pyramid(pyr, levels):
    """
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    display the rendered pyramid
    """
    img = render_pyramid(pyr, levels)
    plt.imshow(img, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    L1, filterL1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, filterL2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    Gm, filterG = build_gaussian_pyramid(mask.astype('float64'), max_levels, filter_size_mask)

    Lout = []
    for k in range(len(Gm)):
        Lout_k = np.multiply(Gm[k], L1[k]) + np.multiply((1-Gm[k]), L2[k])
        Lout.append(Lout_k)
    coeff = np.ones(len(Lout))
    filter_vec = calc_filter_vec(filter_size_im)
    back_to_image = laplacian_to_image(Lout, filter_vec, coeff)
    clip_image = np.clip(back_to_image, 0, 1)
    return clip_image


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    frames = imread(filename).astype('float64')
    if representation == 1:
        if len(frames.shape) == 2:
            return np.divide(frames, 255.0)
        else:
            return skimage.color.rgb2gray(frames)
    else:
        return np.divide(frames, 255.0)


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath("external/river.jpg"), 2)
    im2 = read_image(relpath("external/train.jpg"), 2)
    im = np.zeros((im1.shape[0], im1.shape[1], im1.shape[2]))
    r1 = im1[:, :, 0]
    g1 = im1[:, :, 1]
    b1 = im1[:, :, 2]

    r2 = im2[:, :, 0]
    g2 = im2[:, :, 1]
    b2 = im2[:, :, 2]

    mask = read_image(relpath("external/mask_train.jpg"), 1).astype(bool)
    filter_size_im = 7
    filter_size_mask = 7
    max_levels = 7

    blend_r = pyramid_blending(r1, r2, mask, max_levels, filter_size_im, filter_size_mask)
    blend_g = pyramid_blending(g1, g2, mask, max_levels, filter_size_im, filter_size_mask)
    blend_b = pyramid_blending(b1, b2, mask, max_levels, filter_size_im, filter_size_mask)

    im[:, :, 0] = blend_r
    im[:, :, 1] = blend_g
    im[:, :, 2] = blend_b

    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(im)
    plt.show()

    return im1, im2, mask, im


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath("external/cloud.jpg"), 2)
    im2 = read_image(relpath("external/simpson_new.jpg"), 2)

    im = np.zeros((im1.shape[0], im1.shape[1], im1.shape[2]))
    r1 = im1[:, :, 0]
    g1 = im1[:, :, 1]
    b1 = im1[:, :, 2]

    r2 = im2[:, :, 0]
    g2 = im2[:, :, 1]
    b2 = im2[:, :, 2]

    mask = read_image(relpath("external/mask_new.jpg"), 1).astype(bool)

    filter_size_im = 7
    filter_size_mask = 7
    max_levels = 7

    blend_r = pyramid_blending(r1, r2, mask, max_levels, filter_size_im, filter_size_mask)
    blend_g = pyramid_blending(g1, g2, mask, max_levels, filter_size_im, filter_size_mask)
    blend_b = pyramid_blending(b1, b2, mask, max_levels, filter_size_im, filter_size_mask)

    im[:, :, 0] = blend_r
    im[:, :, 1] = blend_g
    im[:, :, 2] = blend_b

    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(im)
    plt.show()
    return im1, im2, mask, im

if __name__ == '__main__':

    print(np.reshape([1, 1], (1, 2)))
    print(np.array([[1]]))
    print(np.reshape([1], (1, 1)))
