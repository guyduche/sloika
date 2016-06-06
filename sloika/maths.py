import numpy as np

def studentise(x, axis=None):
    """  Studentise a numpy array along a given axis
    :param x: A :class:`ndaray`
    :param axis: axis over which to studentise

    :returns: A :class:`nd.array` with same shape as x
    """
    m = np.mean(x, axis=axis)
    s = np.std(x, axis=axis)
    s = np.where(s > 0.0, s, 1.0)
    if axis is not None:
        sh = np.array(x.shape, dtype=int)
        sh[axis] = 1
        m = m.reshape(sh)
        s = s.reshape(sh)
    return (x - m) / s
