import numpy as np
import pytest

from examples.h5_handling import get_h5py_data


@pytest.fixture(scope="session")
def flourescent_data():
    data = get_h5py_data(
        "./examples/data/fluorescentlights.h5", "/fluorescentlights/1/on", 0, 2_000_000
    )

    return data


@pytest.fixture(scope="session")
def num_ffts(flourescent_data):
    num_ffts = np.floor(np.sqrt(flourescent_data.size / 2))

    return num_ffts
