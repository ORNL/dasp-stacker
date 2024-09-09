"""Dimensionally Aligned Signal Processing Algorithms"""

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from ipywidgets import HBox, Layout, VBox
from scipy import signal
from skimage.transform import resize

from dasp.algorithms import fasp, gen_fft, hasp_decim, hasp_fixed, hasp_interp

# --------- FASP ---------- #


def fasp_update(
    use_row_norm: bool,
    use_im_norm: bool,
    num_ffts: int,
    data: np.ndarray,
    fig: plt.figure,
    ax: plt.axes,
    color_map: str,
    output: widgets.Output,
):
    """Frequency Aligned Signal Projection Visualization Update

    Args:
        use_row_norm (bool): Whether to normalize across each FFT individually.
        use_im_norm (bool): Whether to normalize the entire final image.
        num_ffts (int): Number of fft slices within the STFT.
        data (np.ndarray): The raw signal.
        fig (pyplot.figure): The figure to be updated.
        ax (pyplot.axes): The axes where the output will be displayed.
        color_map (string): The colormap of the image, defaults to "plasma" if none provided.
        output (widgets.Output): The output to be updated for FASP display

    Note:
        This function is intended to be used in conjuction with a widgets.interact call.

    Returns:
        None.
    """

    # create the FASP array
    fasp_array = fasp(
        use_row_norm=use_row_norm,
        use_im_norm=use_row_norm,
        num_ffts=num_ffts,
        signal=data,
    )

    ax.imshow(fasp_array, aspect="auto", cmap=color_map)

    with output:
        clear_output(wait=True)
        display(fig)


def fasp_viz(raw_signal: np.ndarray, num_ffts: int = 0, color_map="plasma"):
    """Frequency Aligned Signal Projection Visualization

    Args:
        raw_signal (np.ndarray): The raw time domain signal
        num_ffts (int): Number of fft slices within the STFT,
                        determined by calculation if not provided.
        color_map (string): The colormap of the image, defaults to "plasma" if none provided.

    Returns:
        UI: The object containing the visualizations.
    """

    # fasp_update needs to be called multiple times, this is for better readability
    def inner_fasp_update():
        fasp_update(
            use_row_norm=use_row_norm.value,
            use_im_norm=use_im_norm.value,
            num_ffts=num_ffts,
            data=raw_signal,
            fig=fasp_fig,
            ax=fasp_ax,
            color_map=color_map.value,
            output=output,
        )

    # set number of ffts if not provided, or invalid (negative)
    if num_ffts <= 0:
        num_ffts = np.floor(np.sqrt(raw_signal.size / 2))

    # create figure and axis for FASP
    fasp_fig, fasp_ax = plt.subplots()

    # edit cosmetics of the image
    fasp_ax.set_title("FASP")
    fasp_fig.canvas.header_visible = False
    fasp_fig.canvas.footer_visible = False
    fasp_ax.set_xlabel("Frequency")
    fasp_ax.set_ylabel("Time")

    # create widget individually
    use_row_norm = widgets.Checkbox(True, description="normalize row?")
    use_im_norm = widgets.Checkbox(True, description="normalize image?")
    color_map = widgets.Text(
        value="plasma",
        description="color map",
        continuous_update=False,
        layout=Layout(width="75%"),
    )

    # create empty output
    output = widgets.Output()
    controls = VBox(
        [
            widgets.Label(
                "CONTROLS", layout=Layout(display="flex", justify_content="center")
            ),
            use_row_norm,
            use_im_norm,
            color_map,
        ]
    )

    # set UI format
    UI = HBox([controls, output])

    # observe each widget for changes
    use_row_norm.observe(
        lambda x: inner_fasp_update(),
        "value",
    )
    use_im_norm.observe(
        lambda x: inner_fasp_update(),
        "value",
    )
    color_map.observe(
        lambda x: inner_fasp_update(),
        "value",
    )
    color_map.observe(
        lambda x: inner_fasp_update(),
        "value",
    )

    plt.close(fasp_fig)
    inner_fasp_update()

    return UI


# ---------- HASP ---------- #


def hasp_update(
    sample_rate,
    fft,
    use_row_norm,
    use_im_norm,
    freq_center,
    bandwidth,
    max_harmonics,
    scale_image,
    fig,
    ax,
    output: widgets.Output,
    alg_type: str = "fixed",
):
    """
    Updates HASP plots based on the provided input data.

    Args:
        sample_rate: The sample rate used for the HASP plot.
        fft: The Fast Fourier Transform (FFT) of the input signal.
        use_row_norm: Whether to use row normalization for the HASP plot.
        use_im_norm: Whether to use image normalization for the HASP plot.
        freq_center: The frequency center used for the HASP plot.
        bandwidth: The bandwidth used for the HASP plot.
        max_harmonics: The maximum number of harmonics to display.
        scale_image: Whether to scale the image for the HASP plot.
        fig: The figure canvas to use for plotting.
        ax: The axis to use for plotting.
        output: The output widget to display the plot in.
        alg_type: The type of HASP algorithm used (e.g. "Fixed HASP"). Defaults to "fixed".

    Returns:
        None
    """
    if alg_type == "fixed":
        hasp_array = hasp_fixed(
            sample_rate=sample_rate,
            fft=fft,
            use_row_norm=use_row_norm,
            use_im_norm=use_im_norm,
            freq_center=freq_center,
            bandwidth=bandwidth,
            max_harmonics=max_harmonics,
        )
        ax.set_title("FIXED HASP")
    elif alg_type == "decimating":
        hasp_array = hasp_decim(
            sample_rate=sample_rate,
            fft=fft,
            use_row_norm=use_row_norm,
            use_im_norm=use_im_norm,
            freq_center=freq_center,
            bandwidth=bandwidth,
            max_harmonics=max_harmonics,
        )
        ax.set_title("DECIMATING HASP")
    elif alg_type == "interpolating":
        hasp_array = hasp_interp(
            sample_rate=sample_rate,
            fft=fft,
            use_row_norm=use_row_norm,
            use_im_norm=use_im_norm,
            freq_center=freq_center,
            bandwidth=bandwidth,
            max_harmonics=max_harmonics,
        )
        ax.set_title("INTERPOLATING HASP")
    else:
        raise ValueError(
            "Invalid HASP type - options are 'fixed', 'decimating', 'interpolating'"
        )

    if scale_image:
        hasp_array = im_scale(
            im=hasp_array, num_bits=16, use_log=True, im_size=[2048, 2048]
        )

    # show image
    ax.imshow(hasp_array, aspect="auto", cmap="plasma", interpolation="none")

    with output:
        clear_output(wait=True)
        display(fig)


def hasp_viz(signal, freq_center, sample_rate):
    """
    Visualizes the HASP algorithm(s).

    Creates an interactive visualization of the HASP algorithm(s) using the provided input data.

    Args:
        signal: The input signal to visualize.
        freq_center: The frequency center of the input signal.
        sample_rate: The sample rate of the input signal.

    Returns:
        UI: The object containing the visualizations.
    """
    # create FFT, needed for hasp_fixed call
    fft = gen_fft(signal)

    # bandwidth slider parameters
    start_bw = freq_center
    max_bw = freq_center
    min_bw = freq_center / 10

    # frequency center slider parameters
    freq_max = freq_center + (freq_center / 25)
    freq_min = freq_center - (freq_center / 25)

    # needs to be called multiple times, this is for better readability
    def inner_hasp_update():
        hasp_update(
            sample_rate=sample_rate,
            fft=fft,
            use_row_norm=use_row_norm.value,
            use_im_norm=use_im_norm.value,
            scale_image=scale_image.value,
            freq_center=freq_center.value,
            bandwidth=bandwidth.value,
            max_harmonics=100,  # max_harmonics,
            fig=hasp_fig,
            ax=hasp_ax,
            output=output,
            alg_type=alg_type.value,
        )

    hasp_fig, hasp_ax = plt.subplots()

    # edit cosmetics
    hasp_ax.set_title("FIXED HASP")
    hasp_fig.canvas.header_visible = False
    hasp_fig.canvas.footer_visible = False
    hasp_ax.set_xlabel("Bin")
    hasp_ax.set_ylabel("Harmonic")

    # create widgets individually
    use_row_norm = widgets.Checkbox(True, description="normalize row?")
    use_im_norm = widgets.Checkbox(True, description="normalize image?")
    scale_image = widgets.Checkbox(True, description="scale image?")
    freq_center = widgets.IntSlider(min=freq_min, max=freq_max, value=freq_center)
    bandwidth = widgets.IntSlider(min=min_bw, max=max_bw, value=start_bw)
    alg_type = widgets.Dropdown(
        options=["fixed", "decimating", "interpolating"],
        value="fixed",
        description="HASP Type:",
    )

    # create empty output
    output = widgets.Output()
    controls = VBox(
        [
            widgets.Label(
                "CONTROLS", layout=Layout(display="flex", justify_content="center")
            ),
            use_row_norm,
            use_im_norm,
            scale_image,
            widgets.Label(
                "frequency center",
                layout=Layout(display="flex", justify_content="center"),
            ),
            freq_center,
            widgets.Label(
                "bandwidth", layout=Layout(display="flex", justify_content="center")
            ),
            bandwidth,
            alg_type,
        ]
    )

    # set UI format
    UI = HBox([controls, output])

    # observe each widget for changes
    use_row_norm.observe(
        lambda x: inner_hasp_update(),
        "value",
    )
    use_im_norm.observe(
        lambda x: inner_hasp_update(),
        "value",
    )
    scale_image.observe(
        lambda x: inner_hasp_update(),
        "value",
    )
    freq_center.observe(
        lambda x: inner_hasp_update(),
        "value",
    )
    bandwidth.observe(
        lambda x: inner_hasp_update(),
        "value",
    )
    alg_type.observe(
        lambda x: inner_hasp_update(),
        "value",
    )

    plt.close(hasp_fig)
    inner_hasp_update()

    return UI


def sine_wave_creator(
    base_freq, sample_rate, duration, noise, wave_type, num_signals, harmonics=0
):
    """
    Creates a sine wave signal with specified characteristics.

    Generates a sine wave signal with the specified base frequency, sample rate, duration,
    and noise level. The wave type can be either 'base', 'square', 'sawtooth', or 'triangle', and the signal can be
    generated with a specified number of harmonics.

    Args:
        base_freq: The base frequency of the sine wave.
        sample_rate: The sample rate of the sine wave.
        duration: The duration of the sine wave.
        noise: The level of noise to add to the sine wave.
        wave_type: The type of sine wave to generate ('base', 'square', 'sawtooth', or 'triangle').
        num_signals: The number of signals to generate.
        harmonics: The number of harmonics to include in the signal (default=0).

    Returns:
        out_signal: The generated sine wave signal.
        time: The associated time array of the generated signal.
    """

    base_freq = base_freq
    sample_rate = sample_rate
    duration = duration
    time = np.linspace(0, duration, duration * sample_rate)
    noise_array = np.random.normal(0, noise, duration * sample_rate)

    base_sig = np.sin(2 * np.pi * base_freq * time) + noise_array

    # add harmonics if applicable, signal + (signal * harmonic #)
    if harmonics != 0:
        for i in range(1, harmonics):
            base_sig = base_sig + np.sin(2 * np.pi * base_freq * time * (i + 1))
    else:
        max_harmonics = (sample_rate // 2) // base_freq
        for i in range(1, max_harmonics):
            base_sig = base_sig + np.sin(2 * np.pi * base_freq * time * (i + 1))

    if wave_type == "square":
        square = signal.square(2 * np.pi * base_freq * time)
        out_signal = square
    elif wave_type == "sawtooth":
        saw = signal.sawtooth(2 * np.pi * base_freq * time)
        out_signal = saw
    elif wave_type == "triangle":
        triangle = signal.sawtooth(2 * np.pi * base_freq * time, width=0.5)
        out_signal = triangle
    else:
        out_signal = base_sig

    # recursively add signals based on the num_signals input parameter
    if num_signals > 1:
        freq_add = int(base_freq + np.sqrt(base_freq))
        other_sig, time = sine_wave_creator(
            freq_add,
            sample_rate,
            duration,
            noise,
            wave_type,
            num_signals - 1,
            harmonics,
        )
        out_signal += other_sig

    return out_signal, time


def sig_update(
    base_freq, sample_rate, duration, harmonics, noise, wave_type, num_signals, fig, ax
):
    """
    Updates the signal plot with the specified parameters.

    This function is used in combination with the sig_viewer function.

    Args:
        base_freq: The base frequency of the sine wave.
        sample_rate: The sample rate of the sine wave.
        duration: The duration of the sine wave.
        harmonics: The number of harmonics included in the signal.
        noise: The level of noise to add to the signal.
        wave_type: The type of sine wave to generate ('base', 'square', 'sawtooth', or 'triangle').
        num_signals: The number of signals to generate.
        fig: The figure to plot the signal on.
        ax: The axes to plot the signal on.

    Returns:
        None
    """
    sig, time = sine_wave_creator(
        base_freq=base_freq,
        sample_rate=sample_rate,
        duration=duration,
        harmonics=harmonics,
        noise=noise,
        wave_type=wave_type,
        num_signals=num_signals,
    )

    # edit cosmetics
    ax.clear()
    ax.plot(time, sig)


def sig_viewer(base_freq, sample_rate, duration, harmonics, wave_type, num_signals):
    """
    Generates a signal based on the given inputs and creates an interactive visualization,
    allowing the user to adjust parameters and see how they effect the signal.

    Args:
        base_freq: The base frequency of the signal.
        sample_rate: The sample rate of the signal.
        duration: The duration of the signal.
        harmonics: The number of harmonics in the signal.
        wave_type: The type of sine wave to generate ('base', 'square', 'sawtooth', or 'triangle').
        num_signals: The number of signals to generate.

    Returns:
        None

    Notes:
        This function is not designed to view external signals.
    """
    sig_fig, sig_ax = plt.subplots()
    sig_ax.set_title("SIGNAL")
    sig_fig.canvas.header_visible = False
    sig_fig.canvas.footer_visible = False
    sig_ax.set_xlabel("Time")
    sig_ax.set_ylabel("Amplitude")

    # max harmonics
    max_harmonics = (sample_rate // 2) // base_freq

    if harmonics > max_harmonics:
        harmonics = max_harmonics

    widgets.interact(
        sig_update,
        base_freq=widgets.IntSlider(min=1, max=base_freq * 2, value=base_freq, step=1),
        sample_rate=widgets.IntSlider(
            min=1, max=sample_rate * 2, value=sample_rate, step=1
        ),
        duration=widgets.IntSlider(min=1, max=5, value=num_signals, step=1),
        harmonics=widgets.IntSlider(min=1, max=max_harmonics, value=harmonics, step=1),
        noise=(0, 1, 0.01),
        wave_type=widgets.fixed(wave_type),
        num_signals=widgets.IntSlider(min=1, max=5, value=num_signals, step=1),
        fig=widgets.fixed(sig_fig),
        ax=widgets.fixed(sig_ax),
    )


def im_scale(use_log, im_size, num_bits, im):
    """
    Scales image based on normalization, log, resize, and quantization

    Args:
        im: The image to scale.
        num_bits: The number of bits to use for scaling. Accepted values are 8, 16, or 32. Defaults to 16 if input is outside these values.
        use_log: Whether to use logarithmic scaling.
        im_size: The size of the output image.

    Returns:
        The scaled image.
    """

    # calculate the log10 of the image pixels
    if use_log:
        im = np.log10(im - np.min(im) + 1)

    # resize image to fixed number of x and y pixels
    if im_size is not None:
        im = resize(im, im_size)

    # scale image to fit within dynamic range of fixed number of bits
    im = (2 ** int(num_bits) - 1) * (im - np.min(im)) / (np.max(im) - np.min(im))

    # convert image pixels to unsigned INTs based on quantization
    if num_bits == 8:
        im = np.uint8(im)
    elif num_bits == 32:
        im = np.uint32(im)
    else:
        num_bits = 16
        im = np.uint16(im)
    return im


class HASPDash:
    def __init__(
        self,
        base_freq: int,
        sample_rate: int,
        duration: int,
        noise: float,
        bandwidth: int,
        num_signals: int,
        use_row_norm: bool = False,
        use_im_norm: bool = False,
    ):
        widget_description_style = {"description_width": "initial"}
        # box_layout = Layout(
        #     display="flex",
        #     flex_flow="column",
        #     align_items="stretch",
        #     border="solid",
        #     width="50%",
        # )

        # initialize variables
        self.base_freq = base_freq
        self.sample_rate = sample_rate
        self.duration = duration
        self.noise = noise
        self.use_row_norm = use_row_norm
        self.use_im_norm = use_im_norm
        self.bandwidth = bandwidth
        self.num_signals = num_signals
        # self.freq_sample = sample_rate

        # bandwidth slider parameters
        # start_bw = base_freq  # TODO: idk if this is the best starting point...
        min_bw = 1

        # frequency center slider parameters

        # set harmonics to max_harmonics
        # TODO: potentially allow more, but have a popup when in nyquist freq range
        self.max_harmonics = (sample_rate // 2) // base_freq
        self.harmonics = self.max_harmonics

        # initialize widgets
        # -- signal widgets
        self.base_freq_widget = widgets.IntSlider(
            min=0,
            max=int(base_freq * 2),
            step=1,
            value=self.base_freq,
            description="base frequency",
            style=widget_description_style,
        )
        self.sample_rate_widget = widgets.IntSlider(
            min=0,
            max=int(sample_rate * 2),
            step=1,
            value=self.sample_rate,
            description="sample rate",
            style=widget_description_style,
        )
        self.duration_widget = widgets.IntSlider(
            min=1,
            max=20,
            step=1,
            value=self.duration,
            description="duration",
            style=widget_description_style,
        )
        self.harmonics_widget = widgets.IntSlider(
            min=0,
            max=self.max_harmonics,
            step=1,
            value=self.harmonics,
            description="harmonics",
            style=widget_description_style,
        )
        self.noise_widget = widgets.FloatSlider(
            min=0,
            max=int(1),
            step=0.01,
            value=self.noise,
            description="noise weight",
            style=widget_description_style,
        )
        self.wave_type_widget = widgets.Dropdown(
            options=["sine", "square", "sawtooth", "triangle"],
            value="sine",
            description="wave type",
            style=widget_description_style,
        )
        # -- algorithm widgets
        self.algo_type_widget = widgets.Dropdown(
            options=["Fixed HASP", "Decimating HASP", "Interpolating HASP"],
            value="Fixed HASP",
            description="algorithm",
            style=widget_description_style,
        )
        self.use_row_norm_widget = widgets.Checkbox(
            description="use row normalization?",
            value=self.use_row_norm,
            style=widget_description_style,
        )
        self.use_im_norm_widget = widgets.Checkbox(
            description="use image normalization?",
            value=self.use_im_norm,
            style=widget_description_style,
        )
        self.scale_image_widget = widgets.Checkbox(
            description="use image scaling?",
            value=False,
            style=widget_description_style,
        )
        self.freq_center_widget = widgets.IntSlider(
            description="frequency center",
            min=0,
            max=base_freq * 2,
            value=base_freq,
            style=widget_description_style,
        )
        self.bandwidth_widget = widgets.IntSlider(
            description="bandwidth",
            min=min_bw,
            max=base_freq,
            value=self.bandwidth,
            style=widget_description_style,
        )
        self.num_signals_widget = widgets.IntSlider(
            description="number of signals",
            min=1,
            max=10,
            value=num_signals,
            style=widget_description_style,
        )

        # create figures/axis
        self.signal_fig, self.signal_ax = plt.subplots()
        self.fft_fig, self.fft_ax = plt.subplots()
        self.specto_fig, self.specto_ax = plt.subplots()
        self.algo_fig, self.algo_ax = plt.subplots()

        # create outputs
        self.signal_output = widgets.Output()
        self.fft_output = widgets.Output()
        self.specto_output = widgets.Output()
        self.algo_output = widgets.Output()

        # observe widgets - will call update if a change is observed for any widget
        self.base_freq_widget.observe(self.on_change, ["value"])
        self.sample_rate_widget.observe(self.on_change, ["value"])
        self.duration_widget.observe(self.on_change, ["value"])
        self.harmonics_widget.observe(self.on_change, ["value"])
        self.noise_widget.observe(self.on_change, ["value"])
        self.wave_type_widget.observe(self.on_change, ["value"])
        self.algo_type_widget.observe(self.on_change, ["value"])
        self.use_im_norm_widget.observe(self.on_change, ["value"])
        self.use_row_norm_widget.observe(self.on_change, ["value"])
        self.scale_image_widget.observe(self.on_change, ["value"])
        self.freq_center_widget.observe(self.on_change, ["value"])
        self.bandwidth_widget.observe(self.on_change, ["value"])
        self.num_signals_widget.observe(self.on_change, ["value"])

        # format display
        self.UI = VBox(
            [
                HBox(
                    [
                        VBox(
                            [
                                self.wave_type_widget,
                                self.base_freq_widget,
                                self.sample_rate_widget,
                                self.duration_widget,
                                self.harmonics_widget,
                                self.noise_widget,
                                self.num_signals_widget,
                            ]
                        ),
                        self.signal_output,
                        self.specto_output,
                    ]
                ),
                HBox(
                    [
                        VBox(
                            [
                                self.algo_type_widget,
                                self.use_row_norm_widget,
                                self.use_im_norm_widget,
                                self.scale_image_widget,
                                self.freq_center_widget,
                                self.bandwidth_widget,
                            ]
                        ),
                        self.fft_output,
                        self.algo_output,
                    ]
                ),
            ]
        )

        # generate data and plot
        # signal
        sig, time = sine_wave_creator(
            base_freq=self.base_freq,
            sample_rate=self.sample_rate,
            duration=self.duration,
            harmonics=self.harmonics,
            noise=self.noise,
            wave_type="sine",
            num_signals=self.num_signals,
        )
        # fft
        fft = gen_fft(sig)

        # spectogram
        spec_freq, spec_time, spec = signal.spectrogram(
            sig, fs=sample_rate, nperseg=int(sample_rate / 2), scaling="spectrum"
        )

        # algo
        haspf_array = hasp_fixed(
            sample_rate=self.sample_rate,
            fft=fft,
            use_row_norm=self.use_row_norm,
            use_im_norm=self.use_im_norm,
            freq_center=self.base_freq,
            bandwidth=self.bandwidth,
            max_harmonics=self.max_harmonics,
        )

        # -- FIGURE CREATION

        plt_height = 5
        plt_width = 3

        harmonic_centers = []

        delta_freq = (self.sample_rate / 2) / (fft.size)
        delta_pnts = np.floor(self.bandwidth / delta_freq)
        delta_fc_point = np.floor(self.base_freq / delta_freq)

        for i in range(self.max_harmonics):
            harmonic_centers.append((i + 1) * (delta_fc_point) - 1)

        self.signal_ax.plot(time, sig)
        self.signal_fig.set_figwidth(plt_height)
        self.signal_fig.set_figheight(plt_width)
        self.signal_ax.set_title("SIGNAL")
        self.signal_ax.set_xlabel("time")
        self.signal_ax.set_ylabel("amplitude")
        plt.close(self.signal_fig)

        self.fft_ax.plot(fft)
        self.fft_fig.set_figwidth(plt_height)
        self.fft_fig.set_figheight(plt_width)
        self.fft_ax.set_title("FFT")
        self.fft_ax.set_xlabel("frequency")
        self.fft_ax.set_ylabel("magnitude")
        [self.fft_ax.axvline(_x, linewidth=1, color="red") for _x in harmonic_centers]
        for i in range(self.max_harmonics):
            delta_min = np.floor(((i + 1) * delta_fc_point) - delta_pnts / 2)
            delta_max = np.floor(((i + 1) * delta_fc_point) + delta_pnts / 2)

            # if bandwidth%2 == 0:
            #     delta_min = int(((i + 1) * base_freq) - (bandwidth / 2))+1
            #     delta_max = int(((i + 1) * base_freq) + (bandwidth / 2))
            # else:
            #     delta_min = int(np.ceil(((i + 1) * base_freq) - (bandwidth / 2)))
            #     delta_max = int(np.floor(((i + 1) * base_freq) + (bandwidth / 2)))

            self.fft_ax.axvspan(delta_min - 1, delta_max - 1, color="red", alpha=0.3)
        plt.close(self.fft_fig)

        self.specto_ax.pcolormesh(
            spec_freq, spec_time, spec.T, shading="auto", cmap="plasma"
        )
        self.specto_fig.set_figwidth(plt_height)
        self.specto_fig.set_figheight(plt_width)
        self.specto_ax.set_title("SPECTOGRAM")
        self.specto_ax.set_xlabel("frequency")
        self.specto_ax.set_ylabel("time")
        plt.close(self.specto_fig)

        self.algo_ax.imshow(
            haspf_array, aspect="auto", cmap="plasma", interpolation="none"
        )
        self.algo_fig.set_figwidth(plt_height)
        self.algo_fig.set_figheight(plt_width)
        self.algo_ax.set_title("HASP")
        self.algo_ax.set_xlabel("bin")
        self.algo_ax.set_ylabel("harmonic")
        plt.close(self.algo_fig)

        with self.signal_output:
            clear_output()
            display(self.signal_fig)
        with self.fft_output:
            clear_output()
            display(self.fft_fig)
        with self.specto_output:
            clear_output(wait=True)
            display(self.specto_fig)
        with self.algo_output:
            clear_output(wait=True)
            display(self.algo_fig)

    def recalc(
        self,
        base_freq,
        sample_rate,
        duration,
        harmonics,
        noise,
        wave_type,
        use_row_norm,
        use_im_norm,
        scale_image,
        algo_type,
        freq_center,
        bandwidth,
        num_signals,
    ):
        self.max_harmonics = (sample_rate // 2) // base_freq

        # generate data for plot
        sig, time = sine_wave_creator(
            base_freq=base_freq,
            sample_rate=sample_rate,
            duration=duration,
            harmonics=harmonics,
            noise=noise,
            wave_type=wave_type,
            num_signals=num_signals,
        )
        fft = gen_fft(sig)
        spec_freq, spec_time, spec = signal.spectrogram(
            sig, fs=sample_rate, nperseg=int(sample_rate / 2)
        )

        if algo_type == "Fixed HASP":
            array = hasp_fixed(
                sample_rate=sample_rate,
                fft=fft,
                use_row_norm=use_row_norm,
                use_im_norm=use_im_norm,
                freq_center=freq_center,
                bandwidth=bandwidth,
                max_harmonics=self.max_harmonics,
            )
        elif algo_type == "Decimating HASP":
            array = hasp_decim(
                sample_rate=sample_rate,
                fft=fft,
                use_row_norm=use_row_norm,
                use_im_norm=use_im_norm,
                freq_center=freq_center,
                bandwidth=bandwidth,
                max_harmonics=self.max_harmonics,
            )
        elif algo_type == "Interpolating HASP":
            array = hasp_interp(
                sample_rate=sample_rate,
                fft=fft,
                use_row_norm=use_row_norm,
                use_im_norm=use_im_norm,
                freq_center=freq_center,
                bandwidth=bandwidth,
                max_harmonics=self.max_harmonics,
            )

        if scale_image:
            array = im_scale(im=array, num_bits=16, use_log=True, im_size=[2048, 2048])

        plt_height = 5
        plt_width = 3

        harmonic_centers = []

        delta_freq = (sample_rate / 2) / fft.size
        delta_pnts = np.ceil(bandwidth / delta_freq)
        delta_fc_point = np.ceil(base_freq / delta_freq)

        for i in range(self.max_harmonics):
            harmonic_centers.append((i + 1) * delta_fc_point - 1)

        self.signal_ax.clear()
        self.signal_ax.plot(time, sig)
        self.signal_fig.set_figwidth(plt_height)
        self.signal_fig.set_figheight(plt_width)
        self.signal_ax.set_title("SIGNAL")
        self.signal_ax.set_xlabel("time")
        self.signal_ax.set_ylabel("amplitude")

        self.fft_ax.clear()
        self.fft_ax.plot(fft)
        self.fft_fig.set_figwidth(plt_height)
        self.fft_fig.set_figheight(plt_width)
        self.fft_ax.set_title("FFT")
        self.fft_ax.set_xlabel("frequency")
        self.fft_ax.set_ylabel("magnitude")
        if algo_type == "Fixed HASP":
            [
                self.fft_ax.axvline(_x, linewidth=1, color="red")
                for _x in harmonic_centers
            ]
            for i in range(self.max_harmonics):
                delta_min = np.floor(((i + 1) * delta_fc_point) - delta_pnts / 2)
                delta_max = np.floor(((i + 1) * delta_fc_point) + delta_pnts / 2)
                self.fft_ax.axvspan(
                    delta_min - 1, delta_max - 1, color="red", alpha=0.3
                )

        self.specto_ax.clear()
        self.specto_ax.pcolormesh(spec_freq, spec_time, spec.T, shading="auto")
        self.specto_fig.set_figwidth(plt_height)
        self.specto_fig.set_figheight(plt_width)
        self.specto_ax.set_title("SPECTOGRAM")
        self.specto_ax.set_xlabel("frequency")
        self.specto_ax.set_ylabel("time")

        self.algo_ax.clear()
        self.algo_ax.imshow(array, aspect="auto", cmap="plasma", interpolation="none")
        self.algo_fig.set_figwidth(plt_height)
        self.algo_fig.set_figheight(plt_width)
        self.algo_ax.set_title("HASP")
        self.algo_ax.set_xlabel("bin")
        self.algo_ax.set_ylabel("harmonic")

    def on_change(self, change):
        self.recalc(
            base_freq=self.base_freq_widget.value,
            sample_rate=self.sample_rate_widget.value,
            duration=self.duration_widget.value,
            harmonics=self.harmonics_widget.value,
            noise=self.noise_widget.value,
            wave_type=self.wave_type_widget.value,
            use_row_norm=self.use_row_norm_widget.value,
            use_im_norm=self.use_im_norm_widget.value,
            scale_image=self.scale_image_widget.value,
            algo_type=self.algo_type_widget.value,
            freq_center=self.freq_center_widget.value,
            bandwidth=self.bandwidth_widget.value,
            num_signals=self.num_signals_widget.value,
        )

        with self.signal_output:
            clear_output(wait=True)
            display(self.signal_fig)
        with self.fft_output:
            clear_output(wait=True)
            display(self.fft_fig)
        with self.specto_output:
            clear_output(wait=True)
            display(self.specto_fig)
        with self.algo_output:
            clear_output(wait=True)
            display(self.algo_fig)


def HASP_dash(base_freq, sample_rate, duration, noise, bandwidth, num_signals):
    """
    Creates a dashboard of the signal, spectogram, FFT, and HASP output.

    This dashboard generates a signal based on the given input and does not accept external signals.
    The purpose of this visualization is to provide common signal visualizations alongside the HASP
    results to demonstrate how a variety of parameters effect the HASP algorithm.

    Args:
        base_freq: The base frequency of the sine wave.
        sample_rate: The sample rate of the sine wave.
        duration: The duration of the sine wave.
        noise: The level of noise to add to the sine wave.
        wave_type: The type of sine wave to generate ('base', 'square', 'sawtooth', or 'triangle').
        bandwidth: The bandwidth aroudn the frequency senter to use for the HASP algorithm. To prevent overlap with other harmonics, this value cannot be larger than the base frequency.
        num_signals: The number of signals to generate.
        harmonics: The number of harmonics to include in the signal (default=0).

    Returns:
        A HASPDash object.

    Notes:
        This function creates a dashboard with four plots: a signal plot, an FFT plot,
        a spectrogram plot, and a HASP plot. The dashboard is interactive, allowing
        the user to adjust parameters such as the base frequency, sample rate, and
        number of harmonics.
    """

    dash = HASPDash(
        base_freq=base_freq,
        sample_rate=sample_rate,
        duration=duration,
        noise=noise,
        bandwidth=bandwidth,
        num_signals=num_signals,
    )

    return dash.UI
