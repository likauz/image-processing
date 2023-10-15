import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import scipy.signal as sps
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
import imageio as iio
import skimage
from skimage.color import rgb2gray


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec

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


"""
This function transform a 1D discrete signal to its Fourier representation
param: signal: 1D signal to transform
return: Returns Fourier signal 
"""


def DFT(signal):
    is_col = False  # if is it column
    N = len(signal)
    if len(signal.shape) == 2:
        signal = signal.flatten()
        is_col = True
    rangeN = np.arange(0, N)
    rangeNT = np.reshape(np.arange(0, N), (N, 1))
    w = np.exp((-2 * np.pi * 1j) * rangeNT * rangeN / N)
    dft_signal = np.dot(w, signal)
    if is_col:
        dft_signal = np.reshape(dft_signal, (N, 1))
    return dft_signal.astype(np.complex128)


"""
This function transform a 1D Fourier to its signal representation
param: signal: 1D Fourier to transform
return: Returns its signal representation
"""


def IDFT(signal):
    is_col = False  # if is it column
    N = len(signal)
    if len(signal.shape) == 2:
        signal = signal.flatten()
        is_col = True
    rangeN = np.arange(0, N)
    rangeNT = np.reshape(np.arange(0, N), (N, 1))
    w = np.exp((2 * np.pi * 1j) * rangeNT * rangeN / N)
    idft_signal = np.dot(w, signal)
    if is_col:
        idft_signal = np.reshape(idft_signal, (N, 1))
    return (1/N) * idft_signal

"""
This function transform a 1D discrete signal to its Fourier representation
param: signal: 2D signal to transform
return: Returns Fourier signal 
"""


def DFT2(image):
    is_big = False  # if is it column
    if len(image.shape) == 3:
        a, b, c = image.shape
        image = np.squeeze(image, axis=(2,))
        is_big = True
    M, N = image.shape  # M num of row, N num of col
    dft_row_image = np.zeros((M, N)).astype(complex)
    dft_image = np.zeros((M, N)).astype(complex)
    for i in range(M):
        dft_row_image[i, :] = DFT(image[i, :])
    for j in range(N):
        dft_image[:, j] = DFT(dft_row_image[:, j])
    if is_big ==True:
        dft_image = np.reshape(dft_image, (a, b, c))
    return dft_image.astype(complex)


"""
This function transform a 2D Fourier to its signal representation
param: signal: 2D Fourier to transform
return: Returns its signal representation
"""

def IDFT2(fourier_image):
    is_big = False  # if is it column
    if len(fourier_image.shape) == 3:
        a, b, c = fourier_image.shape
        fourier_image = np.squeeze(fourier_image, axis=(2,))
        is_big = True
    M, N = fourier_image.shape  # M num of row, N num of col
    idft_row_image = np.zeros((M, N)).astype(complex)
    idft_image = np.zeros((M, N)).astype(complex)
    for i in range(M):
        idft_row_image[i, :] = IDFT(fourier_image[i, :])
    for j in range(N):
        idft_image[:, j] = IDFT(idft_row_image[:, j])
    if is_big == True:
        idft_image = np.reshape(idft_image, (a, b, c))
    return idft_image


""""
This function changes the duration of an given audio file by keeping the same samples,
but changing the sample rate written in the file header.
This function saves the audio in a new file called change_rate.wav
param:  filename: a string representing the path to a WAV file
param:  ratio: representing the duration change.
"""


def change_rate(filename, ratio):
    rate, data = wavfile.read(filename)
    wavfile.write('change_rate.wav', int(rate*ratio), data)


""""
This function changes the duration of an given audio file by reducing the number of samples using Fourier. 
This function does not change the sample rate of the given file.
This function saves the audio in a new file called change_samples.wav
param:  filename: a string representing the path to a WAV file
param:  ratio: representing the duration change.
return: return the new sample points according to ratio
"""


def change_samples(filename, ratio):
    rate, data = wavfile.read(filename)
    res = resize(data, ratio)
    wavfile.write('change_samples.wav', int(rate), res)
    return res.astype('float64')


""""
This function change the number of samples by the given ratio.
param:  data: a 1D array that contain the samples of a WAV file
param:  ratio: representing the duration change.
return: return the new sample points according to ratio
"""


def resize(data, ratio):
    dft = DFT(data)
    shift_dft = np.fft.fftshift(dft)

    if ratio == 1 or ratio == 0:
        return data

    # reduction
    if ratio > 1:
        new_size = int(len(data) / ratio)
        left_slice = int(np.ceil((len(data) - new_size) / 2))
        right_slice = int(np.ceil((len(data) + new_size) / 2))
        slice_data = shift_dft[left_slice: right_slice]
    # extend
    else:
        extra = int(len(data) / ratio - len(data))
        add_end = int(np.ceil(extra/2))
        add_first = int(np.floor(extra/2))
        slice_data = np.pad(shift_dft, (add_first, add_end), 'constant', constant_values=(0, 0))

    shift_again = np.fft.ifftshift(slice_data)
    return IDFT(shift_again).astype(data.dtype)


""""

This function speeds up a WAV file, without changing the pitch, using spectrogram scaling.
param:  data: a 1D array that contain the samples of a WAV file
param:  ratio: representing the duration change.
return: return the new sample points according to ratio
"""


def resize_spectrogram(data, ratio):
    spectogram = stft(data)
    new_spec = np.apply_along_axis(resize, 1, spectogram, ratio)
    return istft(new_spec).astype(data.dtype)


""""
This function speedups a WAV file by phase vocoding its spectrogram.
param:  data: a 1D array that contain the samples of a WAV file
param:  ratio: representing the duration change.
return: return the new sample points according to ratio
"""


def resize_vocoder(data, ratio):
    spec = phase_vocoder(stft(data), ratio)
    return istft(spec).astype(data.dtype)


""""
This function computes the magnitude of a image derivatives
param: im: The image to compute the magnitude of the derivative
return: return the magnitude of the derivative
"""


def conv_der(im):

    conv_x = np.reshape([0.5, 0, -0.5], (1, 3))
    conv_y = np.reshape([0.5, 0, -0.5], (3, 1))

    # convolve between the image and [0.5, 0, -0.5]
    dx = sps.convolve2d(im, conv_x, mode='same')

    # convolve between the image and [0.5, 0, -0.5] transpose
    dy = sps.convolve2d(im, conv_y, mode='same')

    # calculate the magnitude
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


""""
This function computes the magnitude of the image derivatives using Fourier transform.
param: im: The image to compute the magnitude of the derivative.
return: return the magnitude of the derivative
"""


def fourier_der(im):
    N, M = im.shape
    fft_im = DFT2(im)
    fft_im_shift = np.fft.fftshift(fft_im)

    # calculate X derivative
    start = np.ceil(-M/2)
    end = np.ceil(M/2)
    u_x = np.arange(start, end)
    fft_x = np.multiply(fft_im_shift, u_x)
    dx = IDFT2(np.fft.ifftshift(fft_x)) * ((2 * np.pi * 1j) / M)

    # calculate Y derivative
    start = np.ceil(-N/2)
    end = np.ceil(N/2)
    u_y = np.arange(start, end)
    fft_y = np.multiply(fft_im_shift.T, u_y)
    dy = IDFT2(np.fft.ifftshift(fft_y.T)) * ((2 * np.pi * 1j) / N)

    # calculate the magnitude
    magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)
    return magnitude

