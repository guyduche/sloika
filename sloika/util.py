import numpy as np

def geometric_prior(n, m, rev=False):
    """ Make a vec

    :param n: length of vector
    :param m: mean of distribution.
    :param rev: Reverse distribution

    :returns: A 1D :class:`ndarray` containing log probabilities
    """
    p = 1.0 / (1.0 + m)
    prior = np.repeat(np.log(p), n)
    prior[1:] += np.arange(1, n) * np.log1p(-p)
    if rev:
        prior = prior[::-1]
    return prior


def is_contiguous(ndarray):
    '''
    See https://docs.scipy.org/doc/numpy/reference/generated/numpy.ascontiguousarray.html
    '''
    return ndarray.flags['C_CONTIGUOUS']
