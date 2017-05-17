import sloika.module_tools as smt


def network(klen, sd, nfeature=1, winlen=11, stride=5, size=128):

    fun = smt.tanh
    init = smt.partial(smt._rn, sd=sd)

    return smt.Serial([smt.Convolution(nfeature, size, winlen, stride, init=init, has_bias=True, fun=fun),

                       smt.Gru(size, size, init=init, has_bias=True, fun=fun),

                       smt.Reverse(smt.Gru(size, size, init=init, has_bias=True, fun=fun)),

                       smt.Softmax(size, smt.nstate(klen), init=init, has_bias=True)

                      ])
