"""H5 File Handling Functions"""

from typing import io

import h5py

# -- H5 FILE LOADING -- #


def h5py_iterator(file: io, prefix: str = ""):
    """Helper function to recursively loop through all paths in HDF5 file

    Args:
        file (h5py.File): An h5py file
        prefix (str): A filepath within the HDF5 file

    Returns:
        filepath (str), h5py.dataset ?? idk, yield is confusing
    """

    # loop through all keys within the file
    for key in file.keys():
        item = file[key]
        path = f"{prefix}/{key}"

        # yield if item is dataset, otherwise recursively continue to next path
        if isinstance(item, h5py.Dataset):
            yield (path, item)
        elif isinstance(item, h5py.Group):
            yield from h5py_iterator(item, path)


def h5py_contents(filepath: str):
    """Display all filepaths and dataset summaries within HDF5 file

    Args:
        filename (str): The path of the HDF5 file

    Returns:
        None
    """

    with h5py.File(filepath, "r") as f:
        for path, dset in h5py_iterator(f):
            print(path, dset)


def get_h5py_data(filepath: str, h5_path: str, start_index: int = 0, end_index=None):
    """Return dataset from an individual path within an HDF5 file

    Args:
        filepath (str): The filepath of the HDF5 file
        h5_path (str): The path of the datset to load within the HDF5 file
        start_index (int): Starting index of what to return from the H5 dataset
        end_index (int): Ending index of what to return from the H5 dataset

    Note:
        Use h5py_contents to display options for filepaths with the HDF5 file

    Returns:
        h5py.dataset"""

    data = []

    with h5py.File(filepath, "r") as f:
        for path, dset in h5py_iterator(f):
            if path == h5_path:
                data = dset[start_index:end_index]
    return data
