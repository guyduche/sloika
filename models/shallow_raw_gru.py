from sloika.module_tools import *


def network(klen, sd, nfeature, winlen, stride):

    n = 128
    fun = tanh
    init = partial(truncated_normal, sd=sd)

    return Serial([Convolution(nfeature, n, winlen, stride, init=init, has_bias=True, fun=fun),

                   Gru(n, n, init=init, has_bias=True, fun=fun),

                   Reverse(Gru(n, n, init=init, has_bias=True, fun=fun)),

                   Softmax(n, nstate(klen), init=init, has_bias=True)

                   ])
