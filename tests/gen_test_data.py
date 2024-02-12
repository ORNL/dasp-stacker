"""This script is exclusively for generating data for testing by using
the functions/variables as seen in the hackathon workbook"""

import h5py
import numpy as np
from scipy import signal
from scipy.ndimage.filters import uniform_filter
from skimage.transform import resize


class Dasp:
    """This is a class for all of the DASP methods"""

    def __init__(
        self,
        fs,
        is_complex,
        num_bits=16,
        use_row_norm=True,
        use_im_norm=True,
        use_log=True,
        im_size=None,
    ):
        """Initializing the class"""
        self.fs = fs
        self.is_complex = is_complex
        self.use_row_norm = use_row_norm
        self.use_im_norm = use_im_norm
        self.num_bits = num_bits
        self.use_log = use_log
        self.im_size = im_size

    def fasp(self, sig, num_ffts):
        """FASP algorithm requires returning a STFT.
        Arguments:
            sig = raw time domain signal
            num_ffts = number of fft slices withing the STFT
        returns:
            faspArray = STFT image"""
        slice_size = np.int(2 * np.floor(sig.size / (2 * num_ffts)))
        fasp_array = np.zeros(shape=(np.int(num_ffts), np.int(slice_size / 2)))
        for ix in range(0, np.int(num_ffts)):
            sig_row = np.abs(
                np.fft.rfft(
                    np.transpose(sig[ix * slice_size : ix * slice_size + slice_size])
                )
            )
            sig_row = sig_row[0:-1]
            if self.use_row_norm:
                sig_row = self.normrow(sig_row)
            fasp_array[ix, :] = sig_row
        if self.use_im_norm:
            fasp_array = self.normim(fasp_array)
        return fasp_array

    def haspf(self, spec, fc, bw, max_harm):
        """HASPF algorithm requires returning a fixed HASP.
        Arguments:
            spec = real spectrum on input time domain signals
            fc = center frequency of interest
            bw = bandwdith around center frequency
            max_harm = maximum harmonics to use of fc.
        returns:
            hasp_array = fixed HASP Array"""
        df = (self.fs / 2) / spec.size
        pnts = np.ceil(bw / df)
        fcp = np.ceil(fc / df)
        kmax = np.floor(spec.size / (fcp + pnts / 2))
        if kmax > max_harm:
            kmax = max_harm
        hasp_array = np.zeros(shape=(np.int(kmax), np.int(pnts)))
        for ix in range(0, np.int(kmax)):
            spec_row = spec[
                np.int(np.floor((ix + 1) * fcp - pnts / 2)) : np.int(
                    np.floor((ix + 1) * fcp + pnts / 2)
                )
            ]
            if self.use_row_norm:
                spec_row = self.normrow(spec_row)
            hasp_array[ix, :] = spec_row
        if self.use_im_norm:
            hasp_array = self.normim(hasp_array)
        return hasp_array

    def haspd(self, spec, fc, bw, max_harm):
        """HASPF algorithm requires returning a decimating HASP.
        Arguments:
            spec = real spectrum on input time domain signals
            fc = center frequency of interest
            bw = bandwdith around center frequency
            max_harm = maximum harmonics to use of fc.
        returns:
            hasp _array = decimating HASP Array"""
        df = (self.fs / 2) / spec.size
        pnts = np.ceil(bw / df)
        fcp = np.ceil(fc / df)
        kmax = np.floor(spec.size / (fcp + pnts / 2))
        if kmax > max_harm:
            kmax = max_harm
        hasp_array = np.zeros(shape=(np.int(kmax), np.int(pnts)))
        for ix in range(0, np.int(kmax)):
            spec_row = signal.resample(
                spec[
                    np.int(np.floor((ix + 1) * fcp - (ix + 1) * pnts / 2)) : np.int(
                        np.floor((ix + 1) * fcp + (ix + 1) * pnts / 2)
                    )
                ],
                np.int(pnts),
            )
            if self.use_row_norm:
                spec_row = self.normrow(spec_row)
            hasp_array[ix, :] = spec_row
        if self.use_im_norm:
            hasp_array = self.normim(hasp_array)
        return hasp_array

    def haspi(self, spec, fc, bw, max_harm):
        """HASPI algorithm requires returning a interpolating HASP.
        Arguments:
            spec = real spectrum on input time domain signals
            fc = center frequency of interest
            bw = bandwdith around center frequency
            max_harm = maximum harmonics to use of fc.
        returns:
            hasp_array = interpolating HASP Array"""
        df = (self.fs / 2) / spec.size
        pnts = np.ceil(bw / df)
        fcp = np.ceil(fc / df)
        kmax = np.floor(spec.size / (fcp + pnts / 2))
        if kmax > max_harm:
            kmax = max_harm
        maxbins = kmax * pnts
        hasp_array = np.zeros(shape=(np.int(kmax), np.int(maxbins)))
        for ix in range(0, np.int(kmax)):
            inds = np.arange(
                np.int(np.floor((ix + 1) * fcp - (ix + 1) * pnts / 2)),
                np.int(np.ceil((ix + 1) * fcp + (ix + 1) * pnts / 2)),
            )
            tmp = spec[inds[0 : np.int((ix + 1) * pnts)]]
            spec_row = signal.resample(tmp, np.int(maxbins))
            if self.use_row_norm:
                spec_row = self.normrow(spec_row)
            hasp_array[ix, :] = spec_row
        if self.use_im_norm:
            hasp_array = self.normim(hasp_array)
        return hasp_array

    def im_scale(self, im):
        """Scales image based on normailization, log, resize, and
        quantization"""
        # Calculate the log10 of the image pixels
        if self.use_log:
            im = np.log10(im - np.min(im) + 1)
        # resize image to fixed number of x and y pixels
        if self.im_size is not None:
            im = resize(im, self.im_size)
        # Scale image to fit within dynamic range of fixed number of bits
        im = (
            (2 ** np.int(self.num_bits) - 1)
            * (im - np.min(im))
            / (np.max(im) - np.min(im))
        )
        # convert image pixels to unsigned INTs based on quatization
        if self.num_bits == 8:
            im = np.uint8(im)
        elif self.num_bits == 32:
            im = np.uint32(im)
        else:
            self.num_bits = 16
            im = np.uint16(im)
        return im

    def gen_spec(self, sig):
        """Generates real spectrum from real input signal"""
        spec = np.abs(np.fft.rfft(sig))
        spec = spec[0:-1]
        return spec

    def normrow(self, row):
        """Calculates ZMUV of input row argument"""
        row = (row - np.mean(row)) / np.var(row, ddof=1)
        return row

    def normim(self, im):
        """Normalizes 2D image with std filter"""
        window_size = 3
        c1 = uniform_filter(im, window_size, mode="reflect")
        c2 = uniform_filter(im * im, window_size, mode="reflect")
        im = np.sqrt(c2 - c1 * c1)
        return im


# ---------- INITIALIZE CLASS AND VARIABLES ---------- #

sig2 = [1, 2, 3, 4, 5]
cnt = 1

filename = "./data/fluorescentlights.h5"
f = h5py.File("./data/fluorescentlights.h5", "r")


def h5py_dataset_iterator(g, prefix=""):
    # This will only ever set the sig2 variable to the first dataset it comes to, leaving as is
    # There is an updated version in new set of dasp functions, use that when working with h5 files
    global sig2
    global cnt
    for key in g.keys():
        item = g[key]
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Dataset):  # test for dataset
            yield (path, item)
            if cnt == 1:
                sig2 = item[0:2000000]
                print(item[0:5])
                cnt = cnt + 1
        elif isinstance(item, h5py.Group):  # test for group (go down)
            yield from h5py_dataset_iterator(item, path)


with h5py.File(filename, "r") as f:
    for path, dset in h5py_dataset_iterator(f):
        print(path, dset)


# Load data file and collection parameters
fs = 2000000
sig = np.array(sig2)
is_complex = False
ts = sig.size / fs
fc = 45000
bw = 80000
max_harm = 100

# Initialize the DASP class
dasp_proc = Dasp(
    fs,
    is_complex,
    num_bits=16,
    use_row_norm=True,
    use_im_norm=True,
    use_log=True,
    im_size=[2048, 2048],
)


# ---------- GENERATE ARRAYS FOR TESTING ---------- #


numFFTs = np.floor(np.sqrt(sig.size / 2))
spec = dasp_proc.gen_spec(sig)

# FASP
fasp_arr = dasp_proc.fasp(sig, numFFTs)
np.save("./data/fasp_flourescent_test1_arr", fasp_arr)

# HASPF
haspf_arr = dasp_proc.haspf(spec, fc, bw, max_harm)
np.save("./data/haspf_flourescent_test1_arr", haspf_arr)

# HASPD
haspd_arr = dasp_proc.haspd(spec, fc, bw, max_harm)
np.save("./data/haspd_flourescent_test1_arr", haspd_arr)

# HASPI
haspi_arr = dasp_proc.haspi(spec, fc, bw, max_harm)
np.save("./data/haspi_flourescent_test1_arr", haspi_arr)
