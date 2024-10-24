import numpy as np
import pytest


@pytest.fixture(scope="session")
def flourescent_data():
    data = np.load("./tests/data/flourescent_data_1_on.npy")

    return data


@pytest.fixture(scope="session")
def num_ffts(flourescent_data):
    num_ffts = np.floor(np.sqrt(flourescent_data.size / 2))

    return num_ffts
