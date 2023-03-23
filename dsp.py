import numpy as np
import matplotlib.pyplot as plt

def blackman_harris_window(N):
    """
    Create a Blackman-Harris Window
    
    Parameters
    ----------
    N: int
        Length of window
    
    Returns
    -------
    ndarray(N): Samples of the window
    """
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    t = np.arange(N)/N
    return a0 - a1*np.cos(2*np.pi*t) + a2*np.cos(4*np.pi*t) - a3*np.cos(6*np.pi*t)

def stft(x, w, h, win_fn=blackman_harris_window):
    """
    Compute the complex Short-Time Fourier Transform (STFT)
    Parameters
    ----------
    x: ndarray(N)
        Full audio clip of N samples
    w: int
        Window length
    h: int
        Hop length
    win_fn: int -> ndarray(N)
        Window function
    
    Returns
    -------
    ndarray(w, nwindows, dtype=np.complex) STFT
    """
    N = len(x)
    nwin = int(np.ceil((N-w)/h))+1
    # Make a 2D array
    # The rows correspond to frequency bins
    # The columns correspond to windows moved forward in time
    S = np.zeros((w, nwin), dtype=np.complex)
    # Loop through all of the windows, and put the fourier
    # transform amplitudes of each window in its own column
    for j in range(nwin):
        # Pull out the audio in the jth window
        xj = x[h*j:h*j+w]
        # Zeropad if necessary
        if len(xj) < w:
            xj = np.concatenate((xj, np.zeros(w-len(xj))))
        # Apply window function
        xj = win_fn(w)*xj
        # Put the fourier transform into S
        S[:, j] = np.fft.fft(xj)
    return S


def get_constellation(S, freq_win, time_win, max_freq, thresh):
    """
    Parameters
    ----------
    S: ndarray(win, n_windows, dtype=float)
        Absolute magnitude short-time fourier transform
    freq_win: int
        Half-height of max window along frequency axis
    time_win: int
        Half-width of max window along time axis
    max_freq: int
        Maximum frequency bin to consider
    thresh: float
        Minimum max amplitude to consider
        
    Returns
    -------
    I: ndarray(N), J: ndarray(N)
        Frequency and time coordinates of constellation
    """
    from scipy.ndimage import maximum_filter
    M = maximum_filter(S, size=(2*freq_win+1, 2*time_win+1))
    S = S[0:max_freq, :]
    M = M[0:max_freq, :]
    B = S == M
    B = B*(S > thresh)

    I = np.arange(B.shape[0])
    J = np.arange(B.shape[1])
    I, J = np.meshgrid(I, J, indexing='ij')
    I = I[B == 1]
    J = J[B == 1]
    return I, J

def get_fingerprints(I, J, width, height, d_center):
    """
    Compute the fingerprints from a constellation

    Parameters
    ----------
    I: ndarray(N)
        Frequency indices of constellation points
    J: ndarray(N)
        Time indices of constellation points
    width: int
        Half-width of fingerprint window
    height: int
        Half-height of fingerprint window
    d_center: int
        Center offset in time of window.  Should be > width

    Returns
    -------
    hashes: ndarray(M)
        Fingerprint hashes
    offsets: ndarray(M)
        Time offsets of each fingerprint
    """
    hashes = []
    offsets = []
    for i, j in zip(I, J):
        di = I-i
        dj = J-j
        in_range = (di > -height)*(di < height)
        in_range = in_range*(dj>d_center-width)*(dj<d_center+width)
        f1 = [i]*np.sum(in_range)
        f2 = I[in_range]
        dw = dj[in_range]
        h = f1 + f2*256 + dw*256*256 # Convert to hash code
        for hi in h:
            hashes.append(hi)
            offsets.append(j)
    hashes = np.array(hashes, dtype=np.uint32)
    offsets = np.array(offsets, dtype=np.uint32)
    return hashes, offsets