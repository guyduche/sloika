import sloika.module_tools as smt

def network(klen, sd, winlen=3, size=64, nfeature=4):
    """ Create standard Nanonet

    :param size: size of hidden smt
    :param nfilter: Number of filters to use (None: normal windowing)
    :param fun: activation function
    :param klen: Length of kmer

    :returns: a `class`:layer.Layer:
    """
    _prn = smt.partial(smt._rn, sd=sd)
    nstate = (smt._NBASE ** klen) + 1
    gru_act = smt.tanh
    insize = nfeature * winlen

    inlayer = smt.Window(winlen)

    fwd1 = smt.Gru(insize, size, init=_prn, has_bias=True, fun=gru_act)
    fwd2 = smt.Gru(size, size, init=_prn, has_bias=True, fun=gru_act)
    fwd_layer = smt.Serial([fwd1, fwd2])

    bwd1 = smt.Gru(insize, size, init=_prn, has_bias=True, fun=gru_act)
    bwd2 = smt.Gru(size, size, init=_prn, has_bias=True, fun=gru_act)
    bwd_layer = smt.Serial([bwd1, bwd2])

    stacked_layer = smt.birnn(fwd_layer, bwd_layer)

    outlayer = smt.Softmax(2 * size, nstate, init=_prn, has_bias=True)

    return smt.Serial([inlayer, stacked_layer, outlayer])
