import theano.tensor as T
#  Some activation functions
#  Many based on M-estimations functions, see
#  http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html


#  Unbounded
def linear(x):
    return x

def softplus(x):
    return T.nnet.softplus(x)

def relu(x):
    return T.nnet.relu(x)

#  Bounded but monotonic
def tanh(x):
    return T.tanh(x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def erf(x):
    return T.erf(x)

def L1mL2(x):
    return x / T.sqrt(1.0 + 0.5 * T.sqr(x))

def Fair(x):
    return x / (1.0 + T.abs_(x) / 1.3998)

def Retu(x):
    """ Rectifying activation followed by Tanh

    Inspired by more biological neural activation, see figure 1
    http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf
    """
    return T.tanh(T.nnet.relu(x))


#  Bounded and redescending
def sin(x):
    return T.sin(x)

def Cauchy(x):
    return x / (1.0 + T.sqr(x / 2.3849))

def GemanMcClure(x):
    return x / T.sqr(1.0 + T.sqr(x))

def Welsh(x):
    return x * T.exp(-T.sqr(x / 2.9846))
