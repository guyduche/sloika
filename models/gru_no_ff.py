from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import sloika.module_tools as smt


def network(klen, sd, nfeature=4, winlen=3, size=64):
    """ GRU Nanonet with no feed-forward layers

    :param klen: Length of kmer
    :param sd: Standard Deviation of initialisation noise
    :param nfeature: Number of features
    :param winlen: Length of window over data
    :param size: size of hidden recurrent layers

    :returns: a `class`:layer.Layer:
    """
    _prn = smt.partial(smt._rn, sd=sd)
    nstate = smt.nstate(klen)
    gru_act = smt.tanh

    insize = nfeature * winlen

    inlayer = smt.Window(winlen)

    fwd1 = smt.Gru(insize, size, init=_prn, has_bias=True, fun=gru_act)
    bwd1 = smt.Gru(insize, size, init=_prn, has_bias=True, fun=gru_act)
    layer1 = smt.birnn(fwd1, bwd1)

    fwd3 = smt.Gru(2 * size, size, init=_prn, has_bias=True, fun=gru_act)
    bwd3 = smt.Gru(2 * size, size, init=_prn, has_bias=True, fun=gru_act)
    layer3 = smt.birnn(fwd3, bwd3)

    outlayer = smt.Softmax(2 * size, nstate, init=_prn, has_bias=True)

    return smt.Serial([inlayer, layer1, layer3, outlayer])
