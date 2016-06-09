import numpy as np
from sloika import sloika_dtype, layers
from sloika.maths import studentise

def from_events(ev):
    """  Create a matrix of features from
    :param ev: A :class:ndrecarray with fields 'mean', 'stdv' and 'length'

    :Returns: A :class:ndarray with studentised features
    """
    nev = len(ev)
    features = np.zeros((nev, 4), dtype=sloika_dtype)
    features[:,0] = ev['mean']
    features[:,1] = ev['stdv']
    features[:,2] = ev['length']
    #  Zero pad delta mean
    features[:,3] = np.fabs(np.ediff1d(ev['mean'], to_end=0))

    features = np.ascontiguousarray(studentise(features, axis=0), dtype=sloika_dtype)
    return features
