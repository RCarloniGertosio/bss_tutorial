import numpy as np


# Harmonic analysis

def fft(X):
    """Computes the 2D Fast Fourier Transform

    The input and the ouput are flattened. A shift of the zero-frequency component to the center of the spectrum is
    performed.

    Parameters
    ----------
    X: np.ndarray
        (p,) or (n,p) float array, flattened input array or stack of n flattened input arrays

    Returns
    -------
    np.ndarray
        (p,) or (n,p) complex array, shifted and flattened 2D FFT transform or stack of n shifted and flattened 2D
         FFT transforms
    """

    if len(np.shape(X)) == 1:
        size = np.int(np.sqrt(len(X)))
        return np.fft.fftshift(np.fft.fft2(np.reshape(X, (size, size)))).flatten()

    n = np.shape(X)[0]
    size = np.int(np.sqrt(np.shape(X)[1]))
    return np.reshape(np.fft.fftshift(np.fft.fft2(np.reshape(X, (n, size, size))), axes=(1, 2)), (n, size**2))


def ifft(Xfft):
    """Computes the inverse 2D Fast Fourier Transform.

    The input and the ouput are flattened. It is assumed that the input has the zero-frequency component shifted to the
    center.

    Parameters
    ----------
    Xfft: np.ndarray
        (p,) or (n,p) complex array, shifted and flattened 2D FFT transform or stack of n shifted and flattened 2D
        FFT transforms

    Returns
    -------
    np.ndarray
        (p,) or (n,p) float array, flattened input array or stack of n flattened input arrays
    """

    if len(np.shape(Xfft)) == 1:
        size = np.int(np.sqrt(len(Xfft)))
        return np.fft.ifft2(np.fft.fftshift(np.reshape(Xfft, (size, size)))).flatten().real

    n = np.shape(Xfft)[0]
    size = np.int(np.sqrt(np.shape(Xfft)[1]))
    return np.reshape(np.fft.ifft2(np.fft.ifftshift(np.reshape(Xfft, (n, size, size)), axes=(1, 2))), (n, size**2)).real


def fftprod(Xfft, filt):
    """Apply a filter.

    Parameters
    ----------
    Xfft: np.ndarray
        (p,) or (n,p) complex array, input array or stack of n input arrays in Fourier space
    filt: np.ndarray
        (p,) or (n,p) float array, filter or stack of n filters (one filter per input array) in Fourier space

    Returns
    -------
    np.ndarray
        (p,) or (n,p) complex array, input array or stack of n input arrays in Fourier space
    """

    return Xfft * filt


def convolve(X, filt):
    """Convolve arrays with filters.

    Parameters
    ----------
    X: np.ndarray
        (p,) or (n,p) float array, input array or stack of n input arrays
    filt: np.ndarray
        (p,) or (n,p) float array, filter or stack of n filters (one filter per input array) in Fourier space

    Returns
    -------
    X: np.ndarray
        (p,) or (n,p) float array, filtered array or stack of n filtered arrays
    """

    return ifft(fftprod(fft(X), filt))


# Wavelet filtering

def spline2(size, f, fc):
    """
    Compute a non-negative 2D spline, with maximum value 1 at the center. The output is flattened.

    Parameters
    ----------
    size: int
        size of the spline
    f: float
        spline parameter
    fc: float
        spline parameter

    Returns
    -------
    np.ndarray
        (size**2,) float array, spline
    """
    xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
    res = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2).flatten()
    res = 2 * f * res / (fc * size)
    res = (3/2)*1/12*(abs(res-2)**3-4*abs(res-1)**3+6*abs(res)**3-4*abs(res+1)**3+abs(res+2)**3)
    return res


def compute_h(size, fc):
    """
    Compute a 2D low-pass filter, with zero-frequency at the center. The output is flattened.

    Parameters
    ----------
    size: int
        size of the filter
    fc: float
        cutoff parameter

    Returns
    -------
    np.ndarray
        (size**2,) float array, filter
    """

    tab1 = spline2(size, 2 * fc, 1)
    tab2 = spline2(size, fc, 1)
    h = tab1 / (tab2 + 1e-6)
    return h


def compute_g(size, fc):
    """
    Compute a 2D high-pass filter, with zero-frequency at the center. The output is flattened.

    Parameters
    ----------
    size: int
        size of the filter
    fc: float
        cutoff parameter

    Returns
    -------
    np.ndarray
        (size**2,) float array, filter
    """

    tab1 = spline2(size, 2 * fc, 1)
    tab2 = spline2(size, fc, 1)
    g = (tab2 - tab1) / (tab2 + 1e-6)
    return g


def get_wt_filters(size, nscales):
    """Compute wavelet filters, with zero-frequency at the center. The output is flattened.

    Parameters
    ----------
    size: int
        size of the filters
    nscales: int
        number of wavelet detail scales

    Returns
    -------
    np.ndarray
        (size**2,nscales+1) float array, filters
    """

    wt_filters = np.ones((size**2, nscales + 1))
    wt_filters[:, 1:] = np.array([compute_h(size, 2**scale) for scale in range(nscales)]).T
    wt_filters[:, :nscales] -= wt_filters[:, 1:(nscales + 1)]
    return wt_filters


def wt_trans(inputs, nscales=3, fft_in=False, fft_out=False):
    """Wavelet transform an array.

    Parameters
    ----------
    inputs: np.ndarray
        (p,) or (n,p) float array, array or stack of n arrays / if fft_in, (p,) or (p,t) complex array, array or
        stack of n arrays in Fourier space
    nscales: int
        number of wavelet detail scales
    fft_in: bool
        inputs is in Fourier space
    fft_out: bool
        output is in Fourier space

    Returns
    -------
    np.ndarray
        (p,nscales+1) or (n,p,scales+1) float or complex array, wavelet transform of the input array or stack of the
        wavelet transforms of the n input arrays (in Fourier space if fft_out)
    """

    dim_inputs = len(np.shape(inputs))
    X = None  # to remove warnings

    if fft_in:
        Xfft = inputs
        if not fft_out:
            X = ifft(Xfft)
    else:
        X = inputs
        Xfft = fft(X)

    if not fft_out:
        l_scale = X.copy()
        if dim_inputs == 1:
            size2 = len(X)
            wts = np.zeros((size2, nscales + 1))
        else:
            size2 = np.shape(X)[1]
            wts = np.zeros((np.shape(X)[0], size2, nscales + 1))
    else:
        l_scale = Xfft.copy()
        if dim_inputs == 1:
            size2 = np.size(Xfft)
            wts = np.zeros((size2, nscales + 1), dtype='complex')
        else:
            size2 = np.size(np.shape(Xfft)[1])
            wts = np.zeros((np.shape(X)[0], size2, nscales + 1), dtype='complex')

    scale = 1
    for j in range(nscales):
        h = compute_h(np.int(np.sqrt(size2)), scale)
        if not fft_out:
            m = ifft(fftprod(Xfft, h))
        else:
            m = fftprod(Xfft, h)
        h_scale = l_scale - m
        l_scale = m
        if dim_inputs == 1:
            wts[:, j] = h_scale
        else:
            wts[:, :, j] = h_scale
        scale *= 2

    if dim_inputs == 1:
        wts[:, nscales] = l_scale
    else:
        wts[:, :, nscales] = l_scale

    return wts


def wt_rec(wts):
    """Reconstruct a wavelet decomposition.

    Parameters
    ----------
    wts: np.ndarray
        (p,nscales+1) or (n,p,scales+1) float array, wavelet transform of an array or stack of the wavelet transforms of
        n arrays

    Returns
    -------
    np.ndarray
        (p,) or (n,p,) float array, reconstructed array or stack of n reconstructed arrays
    """

    return np.sum(wts, axis=-1)


# Miscellaneous

def getidealbeam(size, cutmin=None, cutmax=None):
    """Compute a 2D isotropic beam, with value 1 until a first cutoff frequency and 0 after a second cutoff frequency.
    The transition is computed with a spline.

    Parameters
    ----------
    size: int
        size of the beam
    cutmin: int
        frequency below which filter is 1 (default: int((size)/6))
    cutmax: int
        frequency above which filter is 0 (default: int((size)/2))

    Returns
    -------
    np.ndarray
        (size**2,) float array, filter
    """

    if cutmin is None:
        cutmin = int(size / 6)
    if cutmax is None:
        cutmax = int(size / 2)
    xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2).flatten()
    beam = 2 * (r - cutmin) / (cutmax - cutmin - 1)
    beam = (3/2)*1/12*(abs(beam-2)**3-4*abs(beam-1)**3+6*abs(beam)**3-4*abs(beam+1)**3+abs(beam+2)**3)
    beam[r <= cutmin] = 1
    beam[r >= cutmax] = 0
    return beam

