import numpy as np

from dasp.algorithms import fasp, gen_fft, hasp_decim, hasp_fixed, hasp_interp


def test_fasp_arr_matching_orig(flourescent_data, num_ffts):
    new_arr = fasp(
        use_row_norm=True, use_im_norm=True, signal=flourescent_data, num_ffts=num_ffts
    )
    orig_arr = np.load("./tests/data/fasp_flourescent_test1_arr.npy")

    np.testing.assert_almost_equal(new_arr, orig_arr)


def test_haspf_arr_matching_orig(flourescent_data):
    fft = gen_fft(flourescent_data)

    new_haspf_arr = hasp_fixed(
        sample_rate=2_000_000,
        use_row_norm=False,
        use_im_norm=False,
        fft=fft,
        freq_center=45_000,
        bandwidth=40_000,
        max_harmonics=100,
    )

    orig_haspf_arr = np.load("./tests/data/haspf_flourescent_test1_arr.npy")

    np.testing.assert_almost_equal(new_haspf_arr, orig_haspf_arr)


def test_haspd_arr_matching_orig(flourescent_data):
    spec = gen_fft(flourescent_data)

    new_haspd_arr = hasp_decim(
        sample_rate=2_000_000,
        use_row_norm=True,
        use_im_norm=True,
        fft=spec,
        freq_center=45000,
        bandwidth=40_000,
        max_harmonics=100,
    )

    orig_haspd_arr = np.load("./tests/data/haspd_flourescent_test1_arr.npy")

    np.testing.assert_almost_equal(new_haspd_arr, orig_haspd_arr)


def test_haspi_arr_matching_orig(flourescent_data):
    spec = gen_fft(flourescent_data)

    new_haspi_arr = hasp_interp(
        sample_rate=2_000_000,
        use_row_norm=True,
        use_im_norm=True,
        fft=spec,
        freq_center=45000,
        bandwidth=40_000,
        max_harmonics=100,
    )

    orig_haspi_arr = np.load("./tests/data/haspi_flourescent_test1_arr.npy")

    np.testing.assert_almost_equal(new_haspi_arr, orig_haspi_arr)


# def plot_fn():
#     def _plot(points):
#         plt.plot(points)
#         yield plt.show()
#         plt.close('all')
#     return _plot


# def test_plot_fn(plot_fn):
#     points = [1, 2, 3]
#     plot_fn(points)
#     assert True
