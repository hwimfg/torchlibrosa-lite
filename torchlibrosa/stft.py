import warnings
import math
import argparse

# import librosa
import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LibrosaLite:
    # TODO: implement cache
    @staticmethod
    def filters_get_window(window, Nx, *, fftbins=True):
        """Compute a window function.

        This is a wrapper for `scipy.signal.get_window` that additionally
        supports callable or pre-computed windows.

        Parameters
        ----------
        window : string, tuple, number, callable, or list-like
            The window specification:

            - If string, it's the name of the window function (e.g., `'hann'`)
            - If tuple, it's the name of the window function and any parameters
            (e.g., `('kaiser', 4.0)`)
            - If numeric, it is treated as the beta parameter of the `'kaiser'`
            window, as in `scipy.signal.get_window`.
            - If callable, it's a function that accepts one integer argument
            (the window length)
            - If list-like, it's a pre-computed window of the correct length `Nx`

        Nx : int > 0
            The length of the window

        fftbins : bool, optional
            If True (default), create a periodic window for use with FFT
            If False, create a symmetric window for filter design applications.

        Returns
        -------
        get_window : np.ndarray
            A window of length `Nx` and type `window`

        See Also
        --------
        scipy.signal.get_window

        Notes
        -----
        This function caches at level 10.

        Raises
        ------
        ParameterError
            If `window` is supplied as a vector of length != `n_fft`,
            or is otherwise mis-specified.
        """
        if callable(window):
            return window(Nx)

        elif isinstance(window, (str, tuple)) or np.isscalar(window):
            # TODO: if we add custom window functions in librosa, call them here

            win: np.ndarray = scipy.signal.get_window(window, Nx, fftbins=fftbins)
            return win

        elif isinstance(window, (np.ndarray, list)):
            if len(window) == Nx:
                return np.asarray(window)

            raise Exception(f"Window size mismatch: {len(window):d} != {Nx:d}")
        else:
            raise Exception(f"Invalid window specification: {window!r}")

    @staticmethod
    def util_pad_center(data: np.ndarray, *, size: int, axis: int = -1, **kwargs):
        """Pad an array to a target length along a target axis.

        This differs from `np.pad` by centering the data prior to padding,
        analogous to `str.center`

        Examples
        --------
        >>> # Generate a vector
        >>> data = np.ones(5)
        >>> librosa.util.pad_center(data, size=10, mode='constant')
        array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

        >>> # Pad a matrix along its first dimension
        >>> data = np.ones((3, 5))
        >>> librosa.util.pad_center(data, size=7, axis=0)
        array([[ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.],
            [ 1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.],
            [ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.]])
        >>> # Or its second dimension
        >>> librosa.util.pad_center(data, size=7, axis=1)
        array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
            [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
            [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

        Parameters
        ----------
        data : np.ndarray
            Vector to be padded and centered
        size : int >= len(data) [scalar]
            Length to pad ``data``
        axis : int
            Axis along which to pad and center the data
        **kwargs : additional keyword arguments
            arguments passed to `np.pad`

        Returns
        -------
        data_padded : np.ndarray
            ``data`` centered and padded to length ``size`` along the
            specified axis

        Raises
        ------
        ParameterError
            If ``size < data.shape[axis]``

        See Also
        --------
        numpy.pad
        """
        kwargs.setdefault("mode", "constant")

        n = data.shape[axis]

        lpad = int((size - n) // 2)

        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (lpad, int(size - n - lpad))

        if lpad < 0:
            raise Exception(
                f"Target size ({size:d}) must be at least input size ({n:d})"
            )

        return np.pad(data, lengths, **kwargs)

    # TODO: implement cache
    @staticmethod
    def util_normalize(
        S: np.ndarray,
        *,
        norm=np.inf,
        axis=0,
        threshold=None,
        fill=None,
    ):
        """Normalize an array along a chosen axis.

        Given a norm (described below) and a target axis, the input
        array is scaled so that::

            norm(S, axis=axis) == 1

        For example, ``axis=0`` normalizes each column of a 2-d array
        by aggregating over the rows (0-axis).
        Similarly, ``axis=1`` normalizes each row of a 2-d array.

        This function also supports thresholding small-norm slices:
        any slice (i.e., row or column) with norm below a specified
        ``threshold`` can be left un-normalized, set to all-zeros, or
        filled with uniform non-zero values that normalize to 1.

        Note: the semantics of this function differ from
        `scipy.linalg.norm` in two ways: multi-dimensional arrays
        are supported, but matrix-norms are not.

        Parameters
        ----------
        S : np.ndarray
            The array to normalize

        norm : {np.inf, -np.inf, 0, float > 0, None}
            - `np.inf`  : maximum absolute value
            - `-np.inf` : minimum absolute value
            - `0`    : number of non-zeros (the support)
            - float  : corresponding l_p norm
                See `scipy.linalg.norm` for details.
            - None : no normalization is performed

        axis : int [scalar]
            Axis along which to compute the norm.

        threshold : number > 0 [optional]
            Only the columns (or rows) with norm at least ``threshold`` are
            normalized.

            By default, the threshold is determined from
            the numerical precision of ``S.dtype``.

        fill : None or bool
            If None, then columns (or rows) with norm below ``threshold``
            are left as is.

            If False, then columns (rows) with norm below ``threshold``
            are set to 0.

            If True, then columns (rows) with norm below ``threshold``
            are filled uniformly such that the corresponding norm is 1.

            .. note:: ``fill=True`` is incompatible with ``norm=0`` because
                no uniform vector exists with l0 "norm" equal to 1.

        Returns
        -------
        S_norm : np.ndarray [shape=S.shape]
            Normalized array

        Raises
        ------
        ParameterError
            If ``norm`` is not among the valid types defined above

            If ``S`` is not finite

            If ``fill=True`` and ``norm=0``

        See Also
        --------
        scipy.linalg.norm

        Notes
        -----
        This function caches at level 40.

        Examples
        --------
        >>> # Construct an example matrix
        >>> S = np.vander(np.arange(-2.0, 2.0))
        >>> S
        array([[-8.,  4., -2.,  1.],
            [-1.,  1., -1.,  1.],
            [ 0.,  0.,  0.,  1.],
            [ 1.,  1.,  1.,  1.]])
        >>> # Max (l-infinity)-normalize the columns
        >>> librosa.util.normalize(S)
        array([[-1.   ,  1.   , -1.   ,  1.   ],
            [-0.125,  0.25 , -0.5  ,  1.   ],
            [ 0.   ,  0.   ,  0.   ,  1.   ],
            [ 0.125,  0.25 ,  0.5  ,  1.   ]])
        >>> # Max (l-infinity)-normalize the rows
        >>> librosa.util.normalize(S, axis=1)
        array([[-1.   ,  0.5  , -0.25 ,  0.125],
            [-1.   ,  1.   , -1.   ,  1.   ],
            [ 0.   ,  0.   ,  0.   ,  1.   ],
            [ 1.   ,  1.   ,  1.   ,  1.   ]])
        >>> # l1-normalize the columns
        >>> librosa.util.normalize(S, norm=1)
        array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
            [-0.1  ,  0.167, -0.25 ,  0.25 ],
            [ 0.   ,  0.   ,  0.   ,  0.25 ],
            [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
        >>> # l2-normalize the columns
        >>> librosa.util.normalize(S, norm=2)
        array([[-0.985,  0.943, -0.816,  0.5  ],
            [-0.123,  0.236, -0.408,  0.5  ],
            [ 0.   ,  0.   ,  0.   ,  0.5  ],
            [ 0.123,  0.236,  0.408,  0.5  ]])

        >>> # Thresholding and filling
        >>> S[:, -1] = 1e-308
        >>> S
        array([[ -8.000e+000,   4.000e+000,  -2.000e+000,
                1.000e-308],
            [ -1.000e+000,   1.000e+000,  -1.000e+000,
                1.000e-308],
            [  0.000e+000,   0.000e+000,   0.000e+000,
                1.000e-308],
            [  1.000e+000,   1.000e+000,   1.000e+000,
                1.000e-308]])

        >>> # By default, small-norm columns are left untouched
        >>> librosa.util.normalize(S)
        array([[ -1.000e+000,   1.000e+000,  -1.000e+000,
                1.000e-308],
            [ -1.250e-001,   2.500e-001,  -5.000e-001,
                1.000e-308],
            [  0.000e+000,   0.000e+000,   0.000e+000,
                1.000e-308],
            [  1.250e-001,   2.500e-001,   5.000e-001,
                1.000e-308]])
        >>> # Small-norm columns can be zeroed out
        >>> librosa.util.normalize(S, fill=False)
        array([[-1.   ,  1.   , -1.   ,  0.   ],
            [-0.125,  0.25 , -0.5  ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.125,  0.25 ,  0.5  ,  0.   ]])
        >>> # Or set to constant with unit-norm
        >>> librosa.util.normalize(S, fill=True)
        array([[-1.   ,  1.   , -1.   ,  1.   ],
            [-0.125,  0.25 , -0.5  ,  1.   ],
            [ 0.   ,  0.   ,  0.   ,  1.   ],
            [ 0.125,  0.25 ,  0.5  ,  1.   ]])
        >>> # With an l1 norm instead of max-norm
        >>> librosa.util.normalize(S, norm=1, fill=True)
        array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
            [-0.1  ,  0.167, -0.25 ,  0.25 ],
            [ 0.   ,  0.   ,  0.   ,  0.25 ],
            [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
        """
        # Avoid div-by-zero
        if threshold is None:

            def tiny(x):
                x = np.asarray(x)

                # Only floating types generate a tiny
                if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
                    x.dtype, np.complexfloating
                ):
                    dtype = x.dtype
                else:
                    dtype = np.dtype(np.float32)

                return np.finfo(dtype).tiny

            threshold = tiny(S)

        elif threshold <= 0:
            raise Exception(f"threshold={threshold} must be strictly positive")

        if fill not in [None, False, True]:
            raise Exception(f"fill={fill} must be None or boolean")

        if not np.all(np.isfinite(S)):
            raise Exception("Input must be finite")

        # All norms only depend on magnitude, let's do that first
        mag = np.abs(S).astype(float)

        # For max/min norms, filling with 1 works
        fill_norm = 1

        if norm is None:
            return S

        elif norm == np.inf:
            length = np.max(mag, axis=axis, keepdims=True)

        elif norm == -np.inf:
            length = np.min(mag, axis=axis, keepdims=True)

        elif norm == 0:
            if fill is True:
                raise Exception("Cannot normalize with norm=0 and fill=True")

            length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

        elif np.issubdtype(type(norm), np.number) and norm > 0:
            length = np.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)

            if axis is None:
                fill_norm = mag.size ** (-1.0 / norm)
            else:
                fill_norm = mag.shape[axis] ** (-1.0 / norm)

        else:
            raise Exception(f"Unsupported norm: {repr(norm)}")

        # indices where norm is below the threshold
        small_idx = length < threshold

        Snorm = np.empty_like(S)
        if fill is None:
            # Leave small indices un-normalized
            length[small_idx] = 1.0
            Snorm[:] = S / length

        elif fill:
            # If we have a non-zero fill value, we locate those entries by
            # doing a nan-divide.
            # If S was finite, then length is finite (except for small positions)
            length[small_idx] = np.nan
            Snorm[:] = S / length
            Snorm[np.isnan(Snorm)] = fill_norm
        else:
            # Set small values to zero by doing an inf-divide.
            # This is safe (by IEEE-754) as long as S is finite.
            length[small_idx] = np.inf
            Snorm[:] = S / length

        return Snorm

    @staticmethod
    def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
        """Compute the sum-square envelope of a window."""
        n = len(x)
        n_fft = len(win_sq)
        for i in range(n_frames):
            sample = i * hop_length
            x[sample : min(n, sample + n_fft)] += win_sq[
                : max(0, min(n_fft, n - sample))
            ]

    @staticmethod
    def filters_window_sumsquare(
        *,
        window,
        n_frames,
        hop_length=512,
        win_length=None,
        n_fft=2048,
        dtype=np.float32,
        norm=None,
    ):
        """Compute the sum-square envelope of a window function at a given hop length.

        This is used to estimate modulation effects induced by windowing observations
        in short-time Fourier transforms.

        Parameters
        ----------
        window : string, tuple, number, callable, or list-like
            Window specification, as in `get_window`
        n_frames : int > 0
            The number of analysis frames
        hop_length : int > 0
            The number of samples to advance between frames
        win_length : [optional]
            The length of the window function.  By default, this matches ``n_fft``.
        n_fft : int > 0
            The length of each analysis frame.
        dtype : np.dtype
            The data type of the output
        norm : {np.inf, -np.inf, 0, float > 0, None}
            Normalization mode used in window construction.
            Note that this does not affect the squaring operation.

        Returns
        -------
        wss : np.ndarray, shape=``(n_fft + hop_length * (n_frames - 1))``
            The sum-squared envelope of the window function

        Examples
        --------
        For a fixed frame length (2048), compare modulation effects for a Hann window
        at different hop lengths:

        >>> n_frames = 50
        >>> wss_256 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=256)
        >>> wss_512 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=512)
        >>> wss_1024 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=1024)

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(nrows=3, sharey=True)
        >>> ax[0].plot(wss_256)
        >>> ax[0].set(title='hop_length=256')
        >>> ax[1].plot(wss_512)
        >>> ax[1].set(title='hop_length=512')
        >>> ax[2].plot(wss_1024)
        >>> ax[2].set(title='hop_length=1024')
        """
        if win_length is None:
            win_length = n_fft

        n = n_fft + hop_length * (n_frames - 1)
        x = np.zeros(n, dtype=dtype)

        # Compute the squared window at the desired length
        win_sq = LibrosaLite.filters_get_window(window, win_length)
        win_sq = LibrosaLite.util_normalize(win_sq, norm=norm) ** 2
        win_sq = LibrosaLite.util_pad_center(win_sq, size=n_fft)

        # Fill the envelope
        LibrosaLite.__window_ss_fill(x, win_sq, n_frames, hop_length)

        return x

    @staticmethod
    def convert_fft_frequencies(*, sr: float = 22050, n_fft: int = 2048) -> np.ndarray:
        """Alternative implementation of `np.fft.fftfreq`

        Parameters
        ----------
        sr : number > 0 [scalar]
            Audio sampling rate
        n_fft : int > 0 [scalar]
            FFT window size

        Returns
        -------
        freqs : np.ndarray [shape=(1 + n_fft/2,)]
            Frequencies ``(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)``

        Examples
        --------
        >>> librosa.fft_frequencies(sr=22050, n_fft=16)
        array([     0.   ,   1378.125,   2756.25 ,   4134.375,
                5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])
        """
        return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    @staticmethod
    def core_hz_to_mel(frequencies, *, htk: bool = False):
        """Convert Hz to Mels

        Examples
        --------
        >>> librosa.hz_to_mel(60)
        0.9
        >>> librosa.hz_to_mel([110, 220, 440])
        array([ 1.65,  3.3 ,  6.6 ])

        Parameters
        ----------
        frequencies : number or np.ndarray [shape=(n,)] , float
            scalar or array of frequencies
        htk : bool
            use HTK formula instead of Slaney

        Returns
        -------
        mels : number or np.ndarray [shape=(n,)]
            input frequencies in Mels

        See Also
        --------
        mel_to_hz
        """
        frequencies = np.asanyarray(frequencies)

        if htk:
            mels: np.ndarray = 2595.0 * np.log10(1.0 + frequencies / 700.0)
            return mels

        # Fill in the linear part
        f_min = 0.0
        f_sp = 200.0 / 3

        mels = (frequencies - f_min) / f_sp

        # Fill in the log-scale part

        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        if frequencies.ndim:
            # If we have array data, vectorize
            log_t = frequencies >= min_log_hz
            mels[log_t] = (
                min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
            )
        elif frequencies >= min_log_hz:
            # If we have scalar data, heck directly
            mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

        return mels

    @staticmethod
    def core_mel_to_hz(mels, *, htk: bool = False):
        """Convert mel bin numbers to frequencies

        Examples
        --------
        >>> librosa.mel_to_hz(3)
        200.

        >>> librosa.mel_to_hz([1,2,3,4,5])
        array([  66.667,  133.333,  200.   ,  266.667,  333.333])

        Parameters
        ----------
        mels : np.ndarray [shape=(n,)], float
            mel bins to convert
        htk : bool
            use HTK formula instead of Slaney

        Returns
        -------
        frequencies : np.ndarray [shape=(n,)]
            input mels in Hz

        See Also
        --------
        hz_to_mel
        """
        mels = np.asanyarray(mels)

        if htk:
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        if mels.ndim:
            # If we have vector data, vectorize
            log_t = mels >= min_log_mel
            freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
        elif mels >= min_log_mel:
            # If we have scalar data, check directly
            freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

        return freqs

    @staticmethod
    def mel_frequencies(
        n_mels: int = 128,
        *,
        fmin: float = 0.0,
        fmax: float = 11025.0,
        htk: bool = False,
    ) -> np.ndarray:
        """Compute an array of acoustic frequencies tuned to the mel scale.

        The mel scale is a quasi-logarithmic function of acoustic frequency
        designed such that perceptually similar pitch intervals (e.g. octaves)
        appear equal in width over the full hearing range.

        Because the definition of the mel scale is conditioned by a finite number
        of subjective psychoaoustical experiments, several implementations coexist
        in the audio signal processing literature [#]_. By default, librosa replicates
        the behavior of the well-established MATLAB Auditory Toolbox of Slaney [#]_.
        According to this default implementation,  the conversion from Hertz to mel is
        linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
        replicates the Hidden Markov Toolkit [#]_ (HTK) according to the following formula::

            mel = 2595.0 * np.log10(1.0 + f / 700.0).

        The choice of implementation is determined by the ``htk`` keyword argument: setting
        ``htk=False`` leads to the Auditory toolbox implementation, whereas setting it ``htk=True``
        leads to the HTK implementation.

        .. [#] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
            In Proc. International Conference on Acoustics, Speech, and Signal Processing
            (ICASSP), vol. 1, pp. 217-220, 1998.

        .. [#] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
            Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.

        .. [#] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
            Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
            The HTK book, version 3.4. Cambridge University, March 2009.

        See Also
        --------
        hz_to_mel
        mel_to_hz
        librosa.feature.melspectrogram
        librosa.feature.mfcc

        Parameters
        ----------
        n_mels : int > 0 [scalar]
            Number of mel bins.
        fmin : float >= 0 [scalar]
            Minimum frequency (Hz).
        fmax : float >= 0 [scalar]
            Maximum frequency (Hz).
        htk : bool
            If True, use HTK formula to convert Hz to mel.
            Otherwise (False), use Slaney's Auditory Toolbox.

        Returns
        -------
        bin_frequencies : ndarray [shape=(n_mels,)]
            Vector of ``n_mels`` frequencies in Hz which are uniformly spaced on the Mel
            axis.

        Examples
        --------
        >>> librosa.mel_frequencies(n_mels=40)
        array([     0.   ,     85.317,    170.635,    255.952,
                341.269,    426.586,    511.904,    597.221,
                682.538,    767.855,    853.173,    938.49 ,
                1024.856,   1119.114,   1222.042,   1334.436,
                1457.167,   1591.187,   1737.532,   1897.337,
                2071.84 ,   2262.393,   2470.47 ,   2697.686,
                2945.799,   3216.731,   3512.582,   3835.643,
                4188.417,   4573.636,   4994.285,   5453.621,
                5955.205,   6502.92 ,   7101.009,   7754.107,
                8467.272,   9246.028,  10096.408,  11025.   ])

        """
        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = LibrosaLite.core_hz_to_mel(fmin, htk=htk)
        max_mel = LibrosaLite.core_hz_to_mel(fmax, htk=htk)

        mels = np.linspace(min_mel, max_mel, n_mels)

        hz: np.ndarray = LibrosaLite.core_mel_to_hz(mels, htk=htk)
        return hz

    # TODO: implement cache
    @staticmethod
    def filters_mel(
        *,
        sr: float,
        n_fft: int,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax,
        htk: bool = False,
        norm="slaney",
        dtype=np.float32,
    ) -> np.ndarray:
        """Create a Mel filter-bank.

        This produces a linear transformation matrix to project
        FFT bins onto Mel-frequency bins.

        Parameters
        ----------
        sr : number > 0 [scalar]
            sampling rate of the incoming signal

        n_fft : int > 0 [scalar]
            number of FFT components

        n_mels : int > 0 [scalar]
            number of Mel bands to generate

        fmin : float >= 0 [scalar]
            lowest frequency (in Hz)

        fmax : float >= 0 [scalar]
            highest frequency (in Hz).
            If `None`, use ``fmax = sr / 2.0``

        htk : bool [scalar]
            use HTK formula instead of Slaney

        norm : {None, 'slaney', or number} [scalar]
            If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization).

            If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
            See `librosa.util.normalize` for a full description of supported norm values
            (including `+-np.inf`).

            Otherwise, leave all the triangles aiming for a peak value of 1.0

        dtype : np.dtype
            The data type of the output basis.
            By default, uses 32-bit (single-precision) floating point.

        Returns
        -------
        M : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
            Mel transform matrix

        See Also
        --------
        librosa.util.normalize

        Notes
        -----
        This function caches at level 10.

        Examples
        --------
        >>> melfb = librosa.filters.mel(sr=22050, n_fft=2048)
        >>> melfb
        array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
            [ 0.   ,  0.   , ...,  0.   ,  0.   ],
            ...,
            [ 0.   ,  0.   , ...,  0.   ,  0.   ],
            [ 0.   ,  0.   , ...,  0.   ,  0.   ]])

        Clip the maximum frequency to 8KHz

        >>> librosa.filters.mel(sr=22050, n_fft=2048, fmax=8000)
        array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
            [ 0.  ,  0.  , ...,  0.  ,  0.  ],
            ...,
            [ 0.  ,  0.  , ...,  0.  ,  0.  ],
            [ 0.  ,  0.  , ...,  0.  ,  0.  ]])

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> img = librosa.display.specshow(melfb, x_axis='linear', ax=ax)
        >>> ax.set(ylabel='Mel filter', title='Mel filter bank')
        >>> fig.colorbar(img, ax=ax)
        """
        if fmax is None:
            fmax = float(sr) / 2

        # Initialize the weights
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = LibrosaLite.convert_fft_frequencies(sr=sr, n_fft=n_fft)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        mel_f = LibrosaLite.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        if isinstance(norm, str):
            if norm == "slaney":
                # Slaney-style mel is scaled to be approx constant energy per channel
                enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
                weights *= enorm[:, np.newaxis]
            else:
                raise Exception(f"Unsupported norm={norm}")
        else:
            weights = LibrosaLite.util_normalize(weights, norm=norm, axis=-1)

        # Only check weights if f_mel[0] is positive
        if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
            # This means we have an empty channel somewhere
            warnings.warn(
                "Empty filters detected in mel frequency basis. "
                "Some channels will produce empty responses. "
                "Try increasing your sampling rate (and fmax) or "
                "reducing n_mels.",
                stacklevel=2,
            )

        return weights


class DFTBase(nn.Module):
    def __init__(self):
        r"""Base class for DFT and IDFT matrix."""
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W


class DFT(DFTBase):
    def __init__(self, n, norm):
        r"""Calculate discrete Fourier transform (DFT), inverse DFT (IDFT,
        right DFT (RDFT) RDFT, and inverse RDFT (IRDFT.)

        Args:
          n: fft window size
          norm: None | 'ortho'
        """
        super(DFT, self).__init__()

        self.W = self.dft_matrix(n)
        self.inv_W = self.idft_matrix(n)

        self.W_real = torch.Tensor(np.real(self.W))
        self.W_imag = torch.Tensor(np.imag(self.W))
        self.inv_W_real = torch.Tensor(np.real(self.inv_W))
        self.inv_W_imag = torch.Tensor(np.imag(self.inv_W))

        self.n = n
        self.norm = norm

    def dft(self, x_real, x_imag):
        r"""Calculate DFT of a signal.

        Args:
            x_real: (n,), real part of a signal
            x_imag: (n,), imag part of a signal

        Returns:
            z_real: (n,), real part of output
            z_imag: (n,), imag part of output
        """
        z_real = torch.matmul(x_real, self.W_real) - torch.matmul(x_imag, self.W_imag)
        z_imag = torch.matmul(x_imag, self.W_real) + torch.matmul(x_real, self.W_imag)
        # shape: (n,)

        if self.norm is None:
            pass
        elif self.norm == "ortho":
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def idft(self, x_real, x_imag):
        r"""Calculate IDFT of a signal.

        Args:
            x_real: (n,), real part of a signal
            x_imag: (n,), imag part of a signal
        Returns:
            z_real: (n,), real part of output
            z_imag: (n,), imag part of output
        """
        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(
            x_imag, self.inv_W_imag
        )
        z_imag = torch.matmul(x_imag, self.inv_W_real) + torch.matmul(
            x_real, self.inv_W_imag
        )
        # shape: (n,)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == "ortho":
            z_real /= math.sqrt(n)
            z_imag /= math.sqrt(n)

        return z_real, z_imag

    def rdft(self, x_real):
        r"""Calculate right RDFT of signal.

        Args:
            x_real: (n,), real part of a signal
            x_imag: (n,), imag part of a signal

        Returns:
            z_real: (n // 2 + 1,), real part of output
            z_imag: (n // 2 + 1,), imag part of output
        """
        n_rfft = self.n // 2 + 1
        z_real = torch.matmul(x_real, self.W_real[..., 0:n_rfft])
        z_imag = torch.matmul(x_real, self.W_imag[..., 0:n_rfft])
        # shape: (n // 2 + 1,)

        if self.norm is None:
            pass
        elif self.norm == "ortho":
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def irdft(self, x_real, x_imag):
        r"""Calculate IRDFT of signal.

        Args:
            x_real: (n // 2 + 1,), real part of a signal
            x_imag: (n // 2 + 1,), imag part of a signal

        Returns:
            z_real: (n,), real part of output
            z_imag: (n,), imag part of output
        """
        n_rfft = self.n // 2 + 1

        flip_x_real = torch.flip(x_real, dims=(-1,))
        flip_x_imag = torch.flip(x_imag, dims=(-1,))
        # shape: (n // 2 + 1,)

        x_real = torch.cat((x_real, flip_x_real[..., 1 : n_rfft - 1]), dim=-1)
        x_imag = torch.cat((x_imag, -1.0 * flip_x_imag[..., 1 : n_rfft - 1]), dim=-1)
        # shape: (n,)

        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(
            x_imag, self.inv_W_imag
        )
        # shape: (n,)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == "ortho":
            z_real /= math.sqrt(n)

        return z_real


class STFT(DFTBase):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    ):
        r"""PyTorch implementation of STFT with Conv1d. The function has the
        same output as librosa.stft.

        Args:
            n_fft: int, fft window size, e.g., 2048
            hop_length: int, hop length samples, e.g., 441
            win_length: int, window length e.g., 2048
            window: str, window function name, e.g., 'hann'
            center: bool
            pad_mode: str, e.g., 'reflect'
            freeze_parameters: bool, set to True to freeze all parameters. Set
                to False to finetune all parameters.
        """
        super(STFT, self).__init__()

        assert pad_mode in ["constant", "reflect"]

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = n_fft

        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        fft_window = librosa.filters.get_window(window, self.win_length, fftbins=True)

        # Pad the window out to n_fft size.
        fft_window = librosa.util.pad_center(fft_window, n_fft)

        # DFT & IDFT matrix.
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=n_fft,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.conv_imag = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=n_fft,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # Initialize Conv1d weights.
        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0:out_channels] * fft_window[:, None]).T
        )[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0:out_channels] * fft_window[:, None]).T
        )[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        r"""Calculate STFT of batch of signals.

        Args:
            input: (batch_size, data_length), input signals.

        Returns:
            real: (batch_size, 1, time_steps, n_fft // 2 + 1)
            imag: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        x = input[:, None, :]  # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag


def magphase(real, imag):
    r"""Calculate magnitude and phase from real and imag part of signals.

    Args:
        real: tensor, real part of signals
        imag: tensor, imag part of signals

    Returns:
        mag: tensor, magnitude of signals
        cos: tensor, cosine of phases of signals
        sin: tensor, sine of phases of signals
    """
    mag = (real**2 + imag**2) ** 0.5
    cos = real / torch.clamp(mag, 1e-10, np.inf)
    sin = imag / torch.clamp(mag, 1e-10, np.inf)

    return mag, cos, sin


class ISTFT(DFTBase):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
        onnx=False,
        frames_num=None,
        device=None,
    ):
        """PyTorch implementation of ISTFT with Conv1d. The function has the
        same output as librosa.istft.

        Args:
            n_fft: int, fft window size, e.g., 2048
            hop_length: int, hop length samples, e.g., 441
            win_length: int, window length e.g., 2048
            window: str, window function name, e.g., 'hann'
            center: bool
            pad_mode: str, e.g., 'reflect'
            freeze_parameters: bool, set to True to freeze all parameters. Set
                to False to finetune all parameters.
            onnx: bool, set to True when exporting trained model to ONNX. This
                will replace several operations to operators supported by ONNX.
            frames_num: None | int, number of frames of audio clips to be
                inferneced. Only useable when onnx=True.
            device: None | str, device of ONNX. Only useable when onnx=True.
        """
        super(ISTFT, self).__init__()

        assert pad_mode in ["constant", "reflect"]

        if not onnx:
            assert frames_num is None, "When onnx=False, frames_num must be None!"
            assert device is None, "When onnx=False, device must be None!"

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.onnx = onnx

        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = self.n_fft

        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        # Initialize Conv1d modules for calculating real and imag part of DFT.
        self.init_real_imag_conv()

        # Initialize overlap add window for reconstruct time domain signals.
        self.init_overlap_add_window()

        if self.onnx:
            # Initialize ONNX modules.
            self.init_onnx_modules(frames_num, device)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def init_real_imag_conv(self):
        r"""Initialize Conv1d for calculating real and imag part of DFT."""
        self.W = self.idft_matrix(self.n_fft) / self.n_fft

        self.conv_real = nn.Conv1d(
            in_channels=self.n_fft,
            out_channels=self.n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.conv_imag = nn.Conv1d(
            in_channels=self.n_fft,
            out_channels=self.n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        ifft_window = LibrosaLite.filters_get_window(
            self.window, self.win_length, fftbins=True
        )
        # (win_length,)

        # Pad the window to n_fft
        ifft_window = LibrosaLite.util_pad_center(ifft_window, self.n_fft)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W * ifft_window[None, :]).T
        )[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W * ifft_window[None, :]).T
        )[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

    def init_overlap_add_window(self):
        r"""Initialize overlap add window for reconstruct time domain signals."""

        ola_window = LibrosaLite.filters_get_window(
            self.window, self.win_length, fftbins=True
        )
        # (win_length,)

        ola_window = LibrosaLite.util_normalize(ola_window, norm=None) ** 2
        ola_window = LibrosaLite.util_pad_center(ola_window, self.n_fft)
        ola_window = torch.Tensor(ola_window)

        self.register_buffer("ola_window", ola_window)
        # (win_length,)

    def init_onnx_modules(self, frames_num, device):
        r"""Initialize ONNX modules.

        Args:
            frames_num: int
            device: str | None
        """

        # Use Conv1d to implement torch.flip(), because torch.flip() is not
        # supported by ONNX.
        self.reverse = nn.Conv1d(
            in_channels=self.n_fft // 2 + 1,
            out_channels=self.n_fft // 2 - 1,
            kernel_size=1,
            bias=False,
        )

        tmp = np.zeros((self.n_fft // 2 - 1, self.n_fft // 2 + 1, 1))
        tmp[:, 1:-1, 0] = np.array(np.eye(self.n_fft // 2 - 1)[::-1])
        self.reverse.weight.data = torch.Tensor(tmp)
        # (n_fft // 2 - 1, n_fft // 2 + 1, 1)

        # Use nn.ConvTranspose2d to implement torch.nn.functional.fold(),
        # because torch.nn.functional.fold() is not supported by ONNX.
        self.overlap_add = nn.ConvTranspose2d(
            in_channels=self.n_fft,
            out_channels=1,
            kernel_size=(self.n_fft, 1),
            stride=(self.hop_length, 1),
            bias=False,
        )

        self.overlap_add.weight.data = torch.Tensor(
            np.eye(self.n_fft)[:, None, :, None]
        )
        # (n_fft, 1, n_fft, 1)

        if frames_num:
            # Pre-calculate overlap-add window sum for reconstructing signals
            # when using ONNX.
            self.ifft_window_sum = self._get_ifft_window_sum_onnx(frames_num, device)
        else:
            self.ifft_window_sum = []

    def forward(self, real_stft, imag_stft, length):
        r"""Calculate inverse STFT.

        Args:
            real_stft: (batch_size, channels=1, time_steps, n_fft // 2 + 1)
            imag_stft: (batch_size, channels=1, time_steps, n_fft // 2 + 1)
            length: int

        Returns:
            real: (batch_size, data_length), output signals.
        """
        assert real_stft.ndimension() == 4 and imag_stft.ndimension() == 4
        batch_size, _, frames_num, _ = real_stft.shape

        real_stft = real_stft[:, 0, :, :].transpose(1, 2)
        imag_stft = imag_stft[:, 0, :, :].transpose(1, 2)
        # (batch_size, n_fft // 2 + 1, time_steps)

        # Get full stft representation from spectrum using symmetry attribute.
        if self.onnx:
            full_real_stft, full_imag_stft = self._get_full_stft_onnx(
                real_stft, imag_stft
            )
        else:
            full_real_stft, full_imag_stft = self._get_full_stft(real_stft, imag_stft)
        # full_real_stft: (batch_size, n_fft, time_steps)
        # full_imag_stft: (batch_size, n_fft, time_steps)

        # Calculate IDFT frame by frame.
        s_real = self.conv_real(full_real_stft) - self.conv_imag(full_imag_stft)
        # (batch_size, n_fft, time_steps)

        # Overlap add signals in frames to reconstruct signals.
        if self.onnx:
            y = self._overlap_add_divide_window_sum_onnx(s_real, frames_num)
        else:
            y = self._overlap_add_divide_window_sum(s_real, frames_num)
        # y: (batch_size, audio_samples + win_length,)

        y = self._trim_edges(y, length)
        # (batch_size, audio_samples,)

        return y

    def _get_full_stft(self, real_stft, imag_stft):
        r"""Get full stft representation from spectrum using symmetry attribute.

        Args:
            real_stft: (batch_size, n_fft // 2 + 1, time_steps)
            imag_stft: (batch_size, n_fft // 2 + 1, time_steps)

        Returns:
            full_real_stft: (batch_size, n_fft, time_steps)
            full_imag_stft: (batch_size, n_fft, time_steps)
        """
        full_real_stft = torch.cat(
            (real_stft, torch.flip(real_stft[:, 1:-1, :], dims=[1])), dim=1
        )
        full_imag_stft = torch.cat(
            (imag_stft, -torch.flip(imag_stft[:, 1:-1, :], dims=[1])), dim=1
        )

        return full_real_stft, full_imag_stft

    def _get_full_stft_onnx(self, real_stft, imag_stft):
        r"""Get full stft representation from spectrum using symmetry attribute
        for ONNX. Replace several pytorch operations in self._get_full_stft()
        that are not supported by ONNX.

        Args:
            real_stft: (batch_size, n_fft // 2 + 1, time_steps)
            imag_stft: (batch_size, n_fft // 2 + 1, time_steps)

        Returns:
            full_real_stft: (batch_size, n_fft, time_steps)
            full_imag_stft: (batch_size, n_fft, time_steps)
        """

        # Implement torch.flip() with Conv1d.
        full_real_stft = torch.cat((real_stft, self.reverse(real_stft)), dim=1)
        full_imag_stft = torch.cat((imag_stft, -self.reverse(imag_stft)), dim=1)

        return full_real_stft, full_imag_stft

    def _overlap_add_divide_window_sum(self, s_real, frames_num):
        r"""Overlap add signals in frames to reconstruct signals.

        Args:
            s_real: (batch_size, n_fft, time_steps), signals in frames
            frames_num: int

        Returns:
            y: (batch_size, audio_samples)
        """

        output_samples = (s_real.shape[-1] - 1) * self.hop_length + self.win_length
        # (audio_samples,)

        # Overlap-add signals in frames to signals. Ref:
        # asteroid_filterbanks.torch_stft_fb.torch_stft_fb() from
        # https://github.com/asteroid-team/asteroid-filterbanks
        y = torch.nn.functional.fold(
            input=s_real,
            output_size=(1, output_samples),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )
        # (batch_size, 1, 1, audio_samples,)

        y = y[:, 0, 0, :]
        # (batch_size, audio_samples)

        # Get overlap-add window sum to be divided.
        ifft_window_sum = self._get_ifft_window(frames_num)
        # (audio_samples,)

        # Following code is abandaned for divide overlap-add window, because
        # not supported by half precision training and ONNX.
        # min_mask = ifft_window_sum.abs() < 1e-11
        # y[:, ~min_mask] = y[:, ~min_mask] / ifft_window_sum[None, ~min_mask]
        # # (batch_size, audio_samples)

        ifft_window_sum = torch.clamp(ifft_window_sum, 1e-11, np.inf)
        # (audio_samples,)

        y = y / ifft_window_sum[None, :]
        # (batch_size, audio_samples,)

        return y

    def _get_ifft_window(self, frames_num):
        r"""Get overlap-add window sum to be divided.

        Args:
            frames_num: int

        Returns:
            ifft_window_sum: (audio_samlpes,), overlap-add window sum to be
            divided.
        """

        output_samples = (frames_num - 1) * self.hop_length + self.win_length
        # (audio_samples,)

        window_matrix = self.ola_window[None, :, None].repeat(1, 1, frames_num)
        # (batch_size, win_length, time_steps)

        ifft_window_sum = F.fold(
            input=window_matrix,
            output_size=(1, output_samples),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )
        # (1, 1, 1, audio_samples)

        ifft_window_sum = ifft_window_sum.squeeze()
        # (audio_samlpes,)

        return ifft_window_sum

    def _overlap_add_divide_window_sum_onnx(self, s_real, frames_num):
        r"""Overlap add signals in frames to reconstruct signals for ONNX.
        Replace several pytorch operations in
        self._overlap_add_divide_window_sum() that are not supported by ONNX.

        Args:
            s_real: (batch_size, n_fft, time_steps), signals in frames
            frames_num: int

        Returns:
            y: (batch_size, audio_samples)
        """

        s_real = s_real[..., None]
        # (batch_size, n_fft, time_steps, 1)

        # Implement overlap-add with Conv1d, because torch.nn.functional.fold()
        # is not supported by ONNX.
        y = self.overlap_add(s_real)[:, 0, :, 0]
        # y: (batch_size, samples_num)

        if len(self.ifft_window_sum) != y.shape[1]:
            device = s_real.device

            self.ifft_window_sum = self._get_ifft_window_sum_onnx(frames_num, device)
            # (audio_samples,)

        # Use torch.clamp() to prevent from underflow to make sure all
        # operations are supported by ONNX.
        ifft_window_sum = torch.clamp(self.ifft_window_sum, 1e-11, np.inf)
        # (audio_samples,)

        y = y / ifft_window_sum[None, :]
        # (batch_size, audio_samples,)

        return y

    def _get_ifft_window_sum_onnx(self, frames_num, device):
        r"""Pre-calculate overlap-add window sum for reconstructing signals when
        using ONNX.

        Args:
            frames_num: int
            device: str | None

        Returns:
            ifft_window_sum: (audio_samples,)
        """

        ifft_window_sum = LibrosaLite.filters_window_sumsquare(
            window=self.window,
            n_frames=frames_num,
            win_length=self.win_length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        # (audio_samples,)

        ifft_window_sum = torch.Tensor(ifft_window_sum)

        if device:
            ifft_window_sum = ifft_window_sum.to(device)

        return ifft_window_sum

    def _trim_edges(self, y, length):
        r"""Trim audio.

        Args:
            y: (audio_samples,)
            length: int

        Returns:
            (trimmed_audio_samples,)
        """
        # Trim or pad to length
        if length is None:
            if self.center:
                y = y[:, self.n_fft // 2 : -self.n_fft // 2]
        else:
            if self.center:
                start = self.n_fft // 2
            else:
                start = 0

            y = y[:, start : start + length]

        return y


class Spectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        freeze_parameters=True,
    ):
        r"""Calculate spectrogram using pytorch. The STFT is implemented with
        Conv1d. The function has the same output of librosa.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def forward(self, input):
        r"""Calculate spectrogram of input signals.
        Args:
            input: (batch_size, data_length)

        Returns:
            spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real**2 + imag**2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (self.power / 2.0)

        return spectrogram


class LogmelFilterBank(nn.Module):
    def __init__(
        self,
        sr=22050,
        n_fft=2048,
        n_mels=64,
        fmin=0.0,
        fmax=None,
        is_log=True,
        ref=1.0,
        amin=1e-10,
        top_db=80.0,
        freeze_parameters=True,
    ):
        r"""Calculate logmel spectrogram using pytorch. The mel filter bank is
        the pytorch implementation of as librosa.filters.mel
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        if fmax == None:
            fmax = sr // 2

        self.melW = LibrosaLite.filters_mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        ).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        r"""Calculate (log) mel spectrogram from spectrogram.

        Args:
            input: (*, n_fft), spectrogram

        Returns:
            output: (*, mel_bins), (log) mel spectrogram
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)
        # (*, mel_bins)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output

    def power_to_db(self, input):
        r"""Power to db, this function is the pytorch implementation of
        librosa.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise Exception("top_db must be non-negative")
                # raise librosa.util.exceptions.ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(
                log_spec, min=log_spec.max().item() - self.top_db, max=np.inf
            )

        return log_spec


class Enframe(nn.Module):
    def __init__(self, frame_length=2048, hop_length=512):
        r"""Enframe a time sequence. This function is the pytorch implementation
        of librosa.util.frame
        """
        super(Enframe, self).__init__()

        self.enframe_conv = nn.Conv1d(
            in_channels=1,
            out_channels=frame_length,
            kernel_size=frame_length,
            stride=hop_length,
            padding=0,
            bias=False,
        )

        self.enframe_conv.weight.data = torch.Tensor(
            torch.eye(frame_length)[:, None, :]
        )
        self.enframe_conv.weight.requires_grad = False

    def forward(self, input):
        r"""Enframe signals into frames.
        Args:
            input: (batch_size, samples)

        Returns:
            output: (batch_size, window_length, frames_num)
        """
        output = self.enframe_conv(input[:, None, :])
        return output

    def power_to_db(self, input):
        r"""Power to db, this function is the pytorch implementation of
        librosa.power_to_lb.
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise Exception("top_db must be non-negative")
                # raise librosa.util.exceptions.ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(
                log_spec, min=log_spec.max() - self.top_db, max=np.inf
            )

        return log_spec


class Scalar(nn.Module):
    def __init__(self, scalar, freeze_parameters):
        super(Scalar, self).__init__()

        self.scalar_mean = Parameter(torch.Tensor(scalar["mean"]))
        self.scalar_std = Parameter(torch.Tensor(scalar["std"]))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        return (input - self.scalar_mean) / self.scalar_std
