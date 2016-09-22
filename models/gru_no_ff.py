import sloika.module_tools as smt

def network(klen, sd, winlen=3, size=64, nfeature=4):
    """ Create standard Nanonet

    :param winlen: Window size
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
    bwd1 = smt.Gru(insize, size, init=_prn, has_bias=True, fun=gru_act)
    layer1 = smt.birnn(fwd1, bwd1)


    fwd3 = smt.Gru(2 * size, size, init=_prn, has_bias=True, fun=gru_act)
    bwd3 = smt.Gru(2 * size, size, init=_prn, has_bias=True, fun=gru_act)
    layer3 = smt.birnn(fwd3, bwd3)


    outlayer = smt.Softmax(2 * size, nstate, init=_prn, has_bias=True)

    return smt.Serial([inlayer, layer1, layer3, outlayer])
