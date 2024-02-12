# Version of the functions I do not want to get rid of yet

import numpy as np
from dasp_algorithms import std_image_filter_2d, zmuv_row_norm
from scipy import signal


def hasp_fixed_old(
    sample_rate: int,
    use_row_norm: bool,
    use_im_norm: bool,
    spectrum: list[float],
    freq_center: int,
    bandwidth: int,
    max_harmonics: int,
):
    # -- this is the slightly modified version of the true original,
    # -- just updated variable names mainly
    # -- also contains the previous kmax calculation, just in case

    delta_freq = (
        sample_rate / 2
    ) / spectrum.size  # spacing of array to get to the frequency center(s)

    delta_pnts = np.ceil(bandwidth / delta_freq)

    delta_fc_point = np.ceil(
        freq_center / delta_freq
    )  # bin that 1st frequency center is in

    if bandwidth > 2 * freq_center:
        raise ValueError("Bandwidth cannot be greater than 2x frequency center")

    # kmax = np.floor(spectrum.size / (fcp + pnts / 2))
    kmax = np.floor((spectrum.size - delta_pnts / 2) / (delta_fc_point))

    if kmax > max_harmonics:
        kmax = max_harmonics

    hasp_array = np.zeros(shape=(int(kmax), int(delta_pnts)))

    for ix in range(0, int(kmax)):
        spec_row = spectrum[
            int(np.floor((ix + 1) * delta_fc_point - delta_pnts / 2)) : int(
                np.floor((ix + 1) * delta_fc_point + delta_pnts / 2)
            )
        ]

        if use_row_norm:
            spec_row = zmuv_row_norm(spec_row)

        hasp_array[ix, :] = spec_row

    if use_im_norm:
        hasp_array = std_image_filter_2d(hasp_array)

    return hasp_array


def hasp_decim(
    freq_sample,
    use_im_norm,
    use_row_norm,
    spectrum,
    freq_center,
    bandwidth,
    max_harmonics,
):
    df = (freq_sample / 2) / spectrum.size
    pnts = np.ceil(bandwidth / df)
    fcp = np.ceil(freq_center / df)
    kmax = np.floor(spectrum.size / (fcp + pnts / 2))
    if kmax > max_harmonics:
        kmax = max_harmonics
    hasp_array = np.zeros(shape=(int(kmax), int(pnts)))
    for ix in range(0, int(kmax)):
        spec_row = signal.resample(
            spectrum[
                int(np.floor((ix + 1) * fcp - (ix + 1) * pnts / 2)) : int(
                    np.floor((ix + 1) * fcp + (ix + 1) * pnts / 2)
                )
            ],
            int(pnts),
        )
        if use_row_norm:
            spec_row = zmuv_row_norm(spec_row)
        hasp_array[ix, :] = spec_row
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
):
    df = (sample_rate / 2) / fft.size
    pnts = np.ceil(bandwidth / df)
    fcp = np.ceil(freq_center / df)
    kmax = np.floor(fft.size / (fcp + pnts / 2))
    if kmax > max_harmonics:
        kmax = max_harmonics
    maxbins = kmax * pnts
    hasp_array = np.zeros(shape=(int(kmax), int(maxbins)))
    for ix in range(0, int(kmax)):
        inds = np.arange(
            int(np.floor((ix + 1) * fcp - (ix + 1) * pnts / 2)),
            int(np.ceil((ix + 1) * fcp + (ix + 1) * pnts / 2)),
        )
        tmp = fft[inds[0 : int((ix + 1) * pnts)]]
        spec_row = signal.resample(tmp, int(maxbins))
        if use_row_norm:
            spec_row = zmuv_row_norm(spec_row)
        hasp_array[ix, :] = spec_row
    if use_im_norm:
        hasp_array = std_image_filter_2d(hasp_array)

    return hasp_array
