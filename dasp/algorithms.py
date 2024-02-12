"""Dimensionally Aligned Signal Processing Algorithms"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter

# -- STACKER ALGORITHMS -- #


def fasp(
    signal: np.ndarray, num_ffts: int, use_row_norm: bool, use_im_norm: bool
) -> np.ndarray:
    """FASP (Frequency Aligned Signal Projection) algorithm generates an STFT.

    Args:
        signal (np.ndarray): The raw time domain signal.
        num_ffts (int): Number of fft slices within the STFT.
        use_row_norm (bool): Whether to normalize across each FFT individually.
        use_im_norm (bool): Whether to normalize the entire final image. (?)

    Returns:
        np.ndarray: The STFT image.
    """

    # determine the size of each slice based on size of signal and number of fft slices
    slice_size = int(2 * np.floor(signal.size / (2 * num_ffts)))

    # shape/fill zero array (slice_size/2 because using real fft)
    fasp_array = np.zeros(shape=(int(num_ffts), int(slice_size / 2)))

    # iterate through signal, generate fft for each section, stack on top of each other
    for fft_num in range(0, int(num_ffts)):
        # set range for section
        section_start = fft_num * slice_size
        section_end = fft_num * slice_size + slice_size

        sig_row = np.abs(np.fft.rfft(np.transpose(signal[section_start:section_end])))

        # remove final point in row
        sig_row = sig_row[0:-1]

        # normalize row if wanted
        if use_row_norm:
            sig_row = zmuv_row_norm(sig_row)

        # assign to appropriate section within array
        fasp_array[fft_num, :] = sig_row

    # normalize entire image if wanted
    if use_im_norm:
        fasp_array = std_image_filter_2d(fasp_array)

    return fasp_array


def hasp_fixed(
    sample_rate: int,
    fft: np.ndarray,
    bandwidth: int,
    freq_center: int,
    use_row_norm: bool,
    use_im_norm: bool,
    max_harmonics=-1,
) -> np.ndarray:
    """Fixed HASP (Harmonically Aligned Signal Projection) algorithm generates a HASP array.

    Args:
        sample_rate (int): Rate at which the the raw time domain signal was acquired or generated.
        fft (np.ndarray): The Fast Fourier Transform of the signal.
        bandwidth (int): How many hertz to include around the frequency center.
        freq_center: The frequency center of the raw time domain signal.
        use_row_norm (bool): Whether to normalize across each FFT individually.
        use_im_norm (bool): Whether to normalize the entire final image.
        max_harmonics: The upper limit of harmonics to include in final HASP array.

    Returns:
        np.ndarray: The fixed HASP array.
    """

    # adjust step size
    # number of points from a sample in the real frequency domain
    real_sample_rate = sample_rate / 2
    # how many samples (periods)
    fc_multiple = fft.size / real_sample_rate
    # adjust bandwidth size to match input as hertz
    bw_points = int(np.ceil(bandwidth * fc_multiple))
    # index for the frequency center
    fc_index = np.ceil(freq_center * fc_multiple)

    # raise an error if the user input a bandwidth size that would cause overlap
    if bandwidth > freq_center:
        raise ValueError("Bandwidth cannot be greater than frequency center")

    # how many harmonics can we have based on fft and bandwidth size
    max_possible_harmonics = int(np.floor((fft.size - bw_points / 2) / (fc_index)))

    # if the user input more harmonics than what's possible, or no harmonics, default to max
    if (max_harmonics > max_possible_harmonics) | (max_harmonics < 1):
        max_harmonics = max_possible_harmonics

    # shape/fill zero array
    hasp_array = np.zeros(shape=(max_harmonics, bw_points))  # fill zero array

    # itterate through fft, grab points around each frequency center, stack on top of each other
    for point in range(1, max_harmonics + 1):
        # after navigating to the center frequency,
        # get the starting index of what the bandwidth will encompass
        min_bw = int(np.floor((point * fc_index) - (bw_points / 2)))
        # repeat for ending index of bandwidth
        max_bw = int(np.floor((point * fc_index) + (bw_points / 2)))

        # get the individual segment
        fft_row = fft[min_bw:max_bw]

        # normalize row if wanted
        if use_row_norm:
            fft_row = zmuv_row_norm(fft_row)

        # assign to appropriate section within array
        hasp_array[point - 1, :] = fft_row

    # normalize entire image if wanted
    if use_im_norm:
        hasp_array = std_image_filter_2d(hasp_array)

    return hasp_array


def hasp_decim(
    sample_rate: int,
    fft: np.ndarray,
    bandwidth: int,
    freq_center: int,
    use_row_norm: bool,
    use_im_norm: bool,
    max_harmonics=-1,
) -> np.ndarray:
    """Decimating HASP (Harmonically Aligned Signal Projection) algorithm generates a HASP array.

    Args:
        sample_rate (int): Rate at which the the raw time domain signal was acquired or generated.
        fft (np.ndarray): The Fast Fourier Transform of the signal.
        bandwidth (int): How many hertz to include around the frequency center.
        freq_center: The frequency center of the raw time domain signal.
        use_row_norm (bool): Whether to normalize across each FFT individually.
        use_im_norm (bool): Whether to normalize the entire final image.
        max_harmonics: The upper limit of harmonics to include in final HASP array.

    Returns:
        np.ndarray: The decimating HASP array.
    """

    # adjust step size
    # number of points from a sample in the real frequency domain
    real_sample_rate = sample_rate / 2
    # how many samples (periods)
    fc_multiple = fft.size / real_sample_rate
    # adjust bandwidth size to match input as hertz
    bw_points = int(np.ceil(bandwidth * fc_multiple))
    # index for the frequency center
    fc_index = np.ceil(freq_center * fc_multiple)

    # raise an error if the user input a bandwidth size that would cause overlap
    if bandwidth > freq_center:
        raise ValueError("Bandwidth cannot be greater than frequency center")

    # how many harmonics can we have based on fft and bandwidth size
    # max_possible_harmonics = int(np.floor((fft.size - bw_points / 2) / (fc_index)))
    max_possible_harmonics = int(np.floor(fft.size / (fc_index + bw_points / 2)))

    # if the user input more harmonics than what's possible, or no harmonics, default to max
    if (max_harmonics > max_possible_harmonics) | (max_harmonics < 1):
        max_harmonics = max_possible_harmonics

    # shape/fill zero array
    hasp_array = np.zeros(shape=(max_harmonics, bw_points))  # fill zero array

    for point in range(1, max_harmonics + 1):
        # after navigating to the center frequency,
        # get the starting index of what the bandwidth will encompass
        min_bw = int(np.floor((point) * fc_index - (point) * bw_points / 2))
        # repeat for ending index of bandwidth
        max_bw = int(np.floor((point) * fc_index + (point) * bw_points / 2))

        # resample row to minimum bins (this is decimation)
        fft_row = signal.resample(fft[min_bw:max_bw], bw_points)

        # normalize row if wanted
        if use_row_norm:
            fft_row = zmuv_row_norm(fft_row)

        # assign to appropriate section within array
        hasp_array[point - 1, :] = fft_row

    # normalize entire image if wanted
    if use_im_norm:
        hasp_array = std_image_filter_2d(hasp_array)

    return hasp_array


def hasp_interp(
    sample_rate: int,
    fft: np.ndarray,
    bandwidth: int,
    freq_center: int,
    use_row_norm: bool,
    use_im_norm: bool,
    max_harmonics=-1,
) -> np.ndarray:
    """Interpolating HASP (Harmonically Aligned Signal Projection) algorithm generates a HASP array.

    Args:
        sample_rate (int): Rate at which the the raw time domain signal was acquired or generated.
        fft (np.ndarray): The Fast Fourier Transform of the signal.
        bandwidth (int): How many hertz to include around the frequency center.
        freq_center (int): The frequency center of the raw time domain signal.
        use_row_norm (bool): Whether to normalize across each FFT individually.
        use_im_norm (bool): Whether to normalize the entire final image.
        max_harmonics (int): The upper limit of harmonics to include in final HASP array.

    Returns:
        np.ndarray: The interpolating HASP array.
    """

    # adjust step size
    # number of points from a sample in the real frequency domain
    real_sample_rate = sample_rate / 2
    # how many samples (periods)
    fc_multiple = fft.size / real_sample_rate
    # adjust bandwidth size to match input as hertz
    bw_points = int(np.ceil(bandwidth * fc_multiple))
    # index for the frequency center
    fc_index = np.ceil(freq_center * fc_multiple)

    # raise an error if the user input a bandwidth size that would cause overlap
    if bandwidth > freq_center:
        raise ValueError("Bandwidth cannot be greater than frequency center")

    # how many harmonics can we have based on fft and bandwidth size
    # max_possible_harmonics = int(np.floor((fft.size - bw_points / 2) / (fc_index)))
    max_possible_harmonics = int(np.floor(fft.size / (fc_index + bw_points / 2)))

    # if the user input more harmonics than what's possible, or no harmonics, default to max
    if (max_harmonics > max_possible_harmonics) | (max_harmonics < 1):
        max_harmonics = max_possible_harmonics

    # this would be the maximum number of bins/points in the last possible harmonic
    maxbins = int(max_possible_harmonics * bw_points)

    # shape/fill zero array
    hasp_array = np.zeros(shape=(max_harmonics, maxbins))  # fill zero array

    for point in range(1, max_harmonics + 1):
        # get the starting index of what the bandwidth will encompass
        # for interpolating, will grow with the harmonic
        min_bw = int(np.floor((point) * fc_index - (point) * bw_points / 2))
        # repeat for ending index of bandwidth
        max_bw = int(np.floor((point) * fc_index + (point) * bw_points / 2))

        # resample row to maximum bins (this is interpolation)
        fft_row = signal.resample(fft[min_bw:max_bw], maxbins)

        # normalize row if wanted
        if use_row_norm:
            fft_row = zmuv_row_norm(fft_row)

        # assign to appropriate section within array
        hasp_array[point - 1, :] = fft_row

    # normalize entire image if wanted
    if use_im_norm:
        hasp_array = std_image_filter_2d(hasp_array)

    return hasp_array


# -- NORMALIZATION -- #


def zmuv_row_norm(row: np.ndarray) -> np.ndarray:
    """Calculates Zero Mean and Unit Variance (ZMUV) of input row argument

    Args:
        row (np.ndarray): A list of values to be normalized

    Returns:
        np.ndarray: normalized array.

    Note:
        A ZMUV normalized image, in addition to contrast enhancement, helps to speed up model
    """

    # ZMUV = (value - mean) / variance
    row = (row - np.mean(row)) / np.var(row, ddof=1)

    return row


def std_image_filter_2d(image: np.ndarray) -> np.ndarray:
    """Normalizes 2D image with standard deviation filter

    Args:
        image (np.ndarray): The image to be normalized

    Returns:
        np.ndarray: normalized image
    """

    # uniform filter replaces the value of a pixel by the mean value of an
    #   area centered at the pixel
    c1 = uniform_filter(image, mode="reflect")
    c2 = uniform_filter(image * image, mode="reflect")

    # standard deviation = sqrt((sum(x - mean(x))^2)/(n-1))
    image = np.sqrt(c2 - c1 * c1)

    return image


# -- OTHER -- #


def gen_fft(signal: list[float] | np.ndarray) -> np.ndarray:
    """Generates real fft from input signal.

    Args:
        signal (array_like): The raw signal.

    Returns:
        np.ndarray: FFT of signal
    """

    # convert to real FFT using numpy
    fft = np.abs(np.fft.rfft(signal))

    # spec[0] contains the zero-frequency term, which is not needed
    fft = fft[1:]

    return fft
