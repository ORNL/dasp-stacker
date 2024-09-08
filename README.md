# Dimensionally Aligned Signal Projection (DASP)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents
1. [Summary](#summary)
2. [Features/Usage](#features--usage)
    * [DASP Algorithms](#dasp-algorithms)
    * [Visualizations](#visualization-functions)
    * [Other](#other-functions)
3. [Installation](#installation)

## Summary
Dimensionally aligned signal projection (DASP) algorithms are used to analyze fast Fourier transforms (FFTs) and generate visualizations that help focus on the harmonics for specific signals. At a high level, these algorithms extract the FFT segments around each harmonic frequency center, and then align them in equally sized arrays ordered by increasing distance from the base frequency. This allows for a focused view of the harmonic frequencies, which, among other use cases, can enable machine learning algorithms to more easily identify salient patterns. This work seeks to provide an effective open-source implementation of the DASP algorithms as well as functionality to help explore and test how these algorithms work with an interactive dashboard and signal-generation tool.

The DASP library is implemented in Python and contains four types of algorithms for implementing these feature engineering techniques: fixed harmonically aligned signal projection (HASP), decimating HASP, interpolating HASP, and frequency aligned signal projection (FASP). Each algorithm returns a numerical array, which can be visualized as an image. For consistency FASP, which is the terminology used for the short-time Fourier transform, has been implemented as part of the library to provide a similar interface to the STFT of the raw signal. Additionally, the library contains an algorithm to generate artificial signals with basic customizations such as the base frequency, sample rate, duration, number of harmonics, noise, and number of signals.

Finally, the library provides multiple interactive visualizations, each implemented using IPyWidgets and work in a Jupyter environment. A dashboard-style visualization is provided, which contains some common signal-processing visual components (signal, FFT, spectogram) updating in unison with the HASP functions (see Figure 1 below). Seperate from the dashboard, an independent visualization is provided for each of the DASP algorithms as well as the artifical signal generator. These visualizations are included in the library to aid in developing an intuitive understanding how the algorithms are affected by different input signals and parameter selections.

A detailed breakdown of how these algorithms work can be seen in 'example/algorithm_walkthrough'

## Features / Usage

Here are some of the main features of the dasp-stacker library:
* Numerical array output for each algorithm (FASP, Fixed HASP, Decimating HASP, Interpolating HASP).
* Dashboard-style interactive visualization of HASP output alongside other, more common, signal visualization methods.
* Custom sine wave generator.
* Interactive visualization for FASP only.
* Interactive visualization for HASP algorithms (selecting one at a time).
* Interactive visualization for sine wave generation.

The features of this library can be thought of as 2 seperate sections. One being simple to use functions to provide raw output of the DASP algorithms, which is the main function of this library. As an added set of features, several visualizations are provided to help develop an intuitive understanding of the HASP algorithms, as well as an interactive signal generation function.

The following sections are a breakout of the recommended usage for the available features of the dasp-stacker library. An example of actual function calls, along with the output, can be seen in 'examples/dasp_example_notebook.ipynb'.

### DASP Algorithms
<hr>

#### FASP
Generates an STFT.

##### Parameters

* signal (np.ndarray): The raw time domain signal.
* num_ffts (int): Number of fft slices within the STFT.
* use_row_norm (bool): Whether to normalize across each FFT individually.
* use_im_norm (bool): Whether to normalize the entire image.

##### Returns
* The FASP array -- a 2D NumPy array.

##### Example Usage

    fasp(signal=signal, num_ffts=np.floor(np.sqrt(signal.size / 2)), use_row_norm=True, use_im_norm=True)

#### Fixed HASP
Fixed Harmonically Aligned Signal Projection generates a HASP array, using a fixed bandwidth around the frequency center.

##### Parameters
* sample_rate (int): Rate at which the the raw time domain signal was acquired or generated, as samples per second.
* fft (np.ndarray): The Fast Fourier Transform of the signal.
* bandwidth (int): How many hertz to include around the frequency center.
* freq_center: The frequency center of the raw time domain signal, in hertz.
* use_row_norm (bool): Whether to normalize across each FFT individually.
* use_im_norm (bool): Whether to normalize the entire final image.
* max_harmonics: The upper limit of harmonics to include in final HASP array.

##### Returns
* The fixed HASP array -- a 2D NumPy array.

##### Example Usage

    hasp_fixed(
        sample_rate=10_000,
        fft=fft,
        bandwidth=1_000,
        freq_center=1_000,
        use_row_norm=True,
        use_im_norm=True,
        )

#### Decimating HASP
Decimating Harmonically Aligned Signal Projection generates a HASP array, allowing bandwidth around the frequency center to grow as the harmonics increase before downsampling each row to the minimum bandwidth size.

##### Parameters
* sample_rate (int): Rate at which the the raw time domain signal was acquired or generated.
* fft (np.ndarray): The Fast Fourier Transform of the signal.
* bandwidth (int): How many hertz to include around the frequency center.
* freq_center: The frequency center of the raw time domain signal.
* use_row_norm (bool): Whether to normalize across each FFT individually.
* use_im_norm (bool): Whether to normalize the entire final image.
* max_harmonics: The upper limit of harmonics to include in final HASP array.

##### Returns:
* The decimating HASP array -- a 2D NumPy array.

##### Example Usage

    hasp_decim(
        sample_rate=10_000,
        fft=fft,
        bandwidth=1_000,
        freq_center=1_000,
        use_row_norm=True,
        use_im_norm=True,
        )

#### Interpolating HASP
Interpolating Harmonically Aligned Signal Projection generates a HASP array, allowing bandwidth around the frequency center to grow as the harmonics increase before upsampling each row to the maximum bandwidth size.

##### Parameters
* sample_rate (int): Rate at which the the raw time domain signal was acquired or generated.
* fft (np.ndarray): The Fast Fourier Transform of the signal.
* bandwidth (int): How many hertz to include around the frequency center.
* freq_center (int): The frequency center of the raw time domain signal.
* use_row_norm (bool): Whether to normalize across each FFT individually.
* use_im_norm (bool): Whether to normalize the entire final image.
* max_harmonics (int): The upper limit of harmonics to include in final HASP array.

##### Returns
* The interpolating HASP array -- a 2D NumPy array.

##### Example Usage

    hasp_interp(
        sample_rate=10_000,
        fft=fft,
        bandwidth=1_000,
        freq_center=1_000,
        use_row_norm=True,
        use_im_norm=True,
        )

### Visualization Functions

--- IN PROGRESS ---


### Other Functions

--- IN PROGRESS ---

## Installation
```python
pip install dasp-stacker
```
