import numpy as np
from sloika import sloika_dtype
from untangled.maths import studentise

NFEATURES = 4

def from_events(ev, tag='scaled_', normalise=True):
    """  Create a matrix of features from

    :param ev: A :class:`ndrecarray` with fields 'mean', 'stdv' and 'length'
    :param tag: Prefix of which fields to read

    :returns: A :class:`ndarray` with studentised features
    """
    nev = len(ev)
    features = np.zeros((nev, 4), dtype=sloika_dtype)
    features[:,0] = ev[tag + 'mean']
    features[:,1] = ev[tag + 'stdv']
    features[:,2] = ev['length']
    #  Zero pad delta mean
    features[:,3] = np.fabs(np.ediff1d(ev[tag + 'mean'], to_end=0))

    if normalise:
        features = studentise(features, axis=0)

    return np.ascontiguousarray(features, dtype=sloika_dtype)
