import numpy as np
from sloika import sloika_dtype
from untangled.maths import med_mad, studentise


def from_events(ev, tag='scaled_'):
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

    features = np.ascontiguousarray(studentise(features, axis=0), dtype=sloika_dtype)
    return features


def filter_by_slope(position, chunk, time=None, fact=3.0):
    """  Filter chunks using slope of mapping

    Fits a linear regression through the mapping of events and indicates
    whether any regions have a unusual slope.

    :param position: A :class:`ndarray` containing positions mapped to.
    :param chunk:
    :param time: A :class:`ndarray` with time of each event or None.  If None,
    the index of the event is used.
    :param fact: Number of standard deviations after which a slope will be
    considered bad.

    :returns: A :class:`ndarray` contain a boolean of whether chunk is good
    """
    assert time is None or len(position) == len(time)
    nchunk = len(position) // chunk
    chunk_pos = position[chunk * np.arange(nchunk)]
    delta_pos = np.diff(chunk_pos)
    if time is None:
        delta_time = chunk
    else:
        chunk_time = time[chunk * np.arange(nchunk)]
        delta_time = np.diff(chunk_time)

    slope = delta_pos / delta_time
    c, t = med_mad(slope)
    slope -= c
    t *= fact
    return np.logical_and(slope < t, slope > -t)
