import numpy as np

def kalman_filter_1d(data, process_variance=1e-5, measurement_variance=1e-2):
    """
    Simple 1D Kalman filter for denoising a 1D signal.
    Args:
        data: 1D numpy array of measurements
        process_variance: Estimated process variance (Q)
        measurement_variance: Estimated measurement variance (R)
    Returns:
        Filtered 1D numpy array
    """
    n = len(data)
    xhat = np.zeros(n)      # a posteri estimate of x
    P = np.zeros(n)         # a posteri error estimate
    xhatminus = np.zeros(n) # a priori estimate of x
    Pminus = np.zeros(n)    # a priori error estimate
    K = np.zeros(n)         # gain or blending factor

    # initial guesses
    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + process_variance

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + measurement_variance)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat