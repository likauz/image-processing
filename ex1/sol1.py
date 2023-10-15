import numpy
import numpy as np
import imageio as iio
import skimage
from skimage.color import rgb2gray
import matplotlib
from matplotlib import pyplot as plt, pyplot

GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])

YIQ_RGB_TRANSFORMATION_MATRIX = np.array([[1, 0.956, 0.619],
                                          [1, -0.272, -0.647],
                                          [1, -1.106, 1.703]])

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


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """

    plt.imshow(read_image(filename, representation), cmap='gray')
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    imYIQ = np.zeros(shape=imRGB.shape)
    imYIQ[:, :, 0] = imRGB[:, :, 0] * RGB_YIQ_TRANSFORMATION_MATRIX[0][0] + \
                     imRGB[:, :, 1] * RGB_YIQ_TRANSFORMATION_MATRIX[0][1] + \
                     imRGB[:, :, 2] * RGB_YIQ_TRANSFORMATION_MATRIX[0][2]
    imYIQ[:, :, 1] = imRGB[:, :, 0] * RGB_YIQ_TRANSFORMATION_MATRIX[1][0] + \
                     imRGB[:, :, 1] * RGB_YIQ_TRANSFORMATION_MATRIX[1][1] + \
                     imRGB[:, :, 2] * RGB_YIQ_TRANSFORMATION_MATRIX[1][2]
    imYIQ[:, :, 2] = imRGB[:, :, 0] * RGB_YIQ_TRANSFORMATION_MATRIX[2][0] + \
                     imRGB[:, :, 1] * RGB_YIQ_TRANSFORMATION_MATRIX[2][1] + \
                     imRGB[:, :, 2] * RGB_YIQ_TRANSFORMATION_MATRIX[2][2]
    return imYIQ


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    imRGB = np.zeros(shape=imYIQ.shape)
    imRGB[:, :, 0] = imYIQ[:, :, 0] * YIQ_RGB_TRANSFORMATION_MATRIX[0][0] + \
                     imYIQ[:, :, 1] * YIQ_RGB_TRANSFORMATION_MATRIX[0][1] + \
                     imYIQ[:, :, 2] * YIQ_RGB_TRANSFORMATION_MATRIX[0][2]
    imRGB[:, :, 1] = imYIQ[:, :, 0] * YIQ_RGB_TRANSFORMATION_MATRIX[1][0] + \
                     imYIQ[:, :, 1] * YIQ_RGB_TRANSFORMATION_MATRIX[1][1] + \
                     imYIQ[:, :, 2] * YIQ_RGB_TRANSFORMATION_MATRIX[1][2]
    imRGB[:, :, 2] = imYIQ[:, :, 0] * YIQ_RGB_TRANSFORMATION_MATRIX[2][0] + \
                     imYIQ[:, :, 1] * YIQ_RGB_TRANSFORMATION_MATRIX[2][1] + \
                     imYIQ[:, :, 2] * YIQ_RGB_TRANSFORMATION_MATRIX[2][2]
    return imRGB


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """

    if len(im_orig.shape) == 3:  # if it RGB transfer to YIQ
        imYIQ = rgb2yiq(im_orig)
        matrix = imYIQ[:, :, 0]  # Y
    else:
        matrix = im_orig  # grayscale

    matrix = (matrix * 255).astype(int)  # from [0,1] to [0,256]
    hist_orig, x = np.histogram(matrix, bins=256, range=(0, 255))
    C = np.cumsum(hist_orig)
    N = C[255]  # pixels number
    normalize = (255/N) * C
    m = np.argmax(C != 0)
    Cm = normalize[m]  # the number of the min pixel
    T = np.round(((C - Cm)/(C[255]-Cm)) * 255)

    func = lambda x : T[x]
    newMatrix = func(matrix)  # im_eq[i] = T[im_orig[i]]
    newMatrix = newMatrix / 255
    newMatrix = newMatrix.astype('float64')
    hist_eq, x = np.histogram(newMatrix*255, bins=256, range=(0, 255))

    if len(im_orig.shape) == 3:  # if it YIQ transfer to RGB
        imYIQ[:, :, 0] = newMatrix
        im_eq = yiq2rgb(imYIQ)
    else:
        im_eq = newMatrix
    return [im_eq, hist_orig, hist_eq]


def create_image(n_quant, matrix, z, q):
    pixel_hist = np.array(list(range(0, 256)))
    for i in range(n_quant):
        start, end = int(z[i]) + 1, int(z[i + 1]) + 1
        pixel_hist[start: end] = q[i]
    func = lambda v: pixel_hist[v]
    newMatrix = func(matrix.astype(int))
    newMatrix = newMatrix / 255
    return newMatrix.astype('float64')


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    if len(im_orig.shape) == 3:  # if it's RGB transfer to YIQ
        imYIQ = rgb2yiq(im_orig)
        matrix = imYIQ[:, :, 0]  # Y
    else:
        matrix = im_orig  # grayscale
    matrix = (matrix * 255).astype(int)

    hist_orig, x = np.histogram(matrix, bins=256, range=(0, 255))
    cumsum = np.cumsum(hist_orig)

    z = []
    space = cumsum[255]/n_quant
    for v in range(n_quant):
        sub = np.abs(cumsum - v * space)
        z.append(np.argmin(sub))
    z.append(255)
    z[0] = -1
    z = np.array(z)
    q = np.array((z[:-1] + z[1:])/2).astype(int)
    g = list(range(0, 256))
    gh = numpy.multiply(g, hist_orig)  # g * h[g]
    error = []
    newZ = np.zeros(n_quant+1)
    for i in range(n_iter):
        curerror = 0
        # calculate q
        for j in range(n_quant):
            start, end = int(z[j]) + 1, int(z[j+1]) + 1
            q[j] = (np.sum(gh[start: end]))/(np.sum(hist_orig[start: end]))
        # calculate error
        for j in range(n_quant):
            start, end = int(z[j]) + 1, int(z[j + 1]) + 1
            rangeG = (numpy.array(list(range(start, end))))
            pow = numpy.power(q[j] - rangeG, 2)
            curerror += sum(numpy.multiply(pow, hist_orig[start: end]))
        # calculate newZ
        newZ[0] = -1
        for j in range(1, n_quant):
            newZ[j] = (q[j - 1] + q[j]) / 2
        newZ[n_quant] = 255
        # check if equal
        if np.array_equal(newZ, z):
            break
        z = np.copy(newZ)
        error.append(int(curerror))

    newMatrix = create_image(n_quant, matrix, z, q)

    if len(im_orig.shape) == 3:  # if it RGB transfer to YIQ
        imYIQ[:, :, 0] = newMatrix
        im_new = yiq2rgb(imYIQ)
    else:
        im_new = newMatrix
    return [im_new, error]


def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensit ies im_quant image will have
    :return:  im_quant - the quantized output image
    """
    from sklearn.utils import shuffle
    from sklearn.cluster import MiniBatchKMeans

    im_orig = im_orig * 255
    w, h, d = tuple(im_orig.shape)
    image_array = im_orig.reshape((w * h), d)
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = MiniBatchKMeans(n_clusters=n_quant).fit(image_array_sample)  # find the n quant
    labels = kmeans.predict(image_array)
    quantized_image = kmeans.cluster_centers_.astype("uint8")[labels]
    quantized_image = quantized_image.reshape((w, h, d))
    quantized_image = quantized_image/255
    return quantized_image

if __name__ == '__main__':
    im  = read_image('monkey.jpg', 2)
    im = quantize(im, 2, 1000)
    plt.imshow(im[0], cmap ='gray',vmin = 0,vmax =255)
    plt.show()




