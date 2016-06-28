import numpy as np

def argmax(post):
    """  Argmax decoding of simple transducer

    :param post: A 2D :class:`ndarray`

    :returns: A 1D :class:`ndarray` containing called sequence
    """
    blank_state = post.shape[1] - 1
    path = np.argmax(post, axis=1)
    return path[path != blank_state]


def score(post, seq, full=False):
    """  Compute score of a sequence

    :param post: A 2D :class:`ndarray`
    :param seq: Sequence to map against
    :param full: Force full length mapping

    :returns: score
    """
    return forwards(post, seq, full=full)


def refine(post, seq):
    """  Iterative refinement of a sequence

    :param post: A 2D :class:`ndarray`
    :param seq: Initial sequence to map against

    :returns:
    """
    pass


def forwards(post, seq, full=False):
    """ The forwards score for sequence

    :param post: A 2D :class:`ndarray`
    :param seq: Sequence to map against
    :param full: Force full length mapping

    :returns: score
    """
    seq_len = len(seq)

    #  Use seq_len + 1 since additional 'blank state' at beginning
    fwd = np.ones(seq_len + 1)
    fprev = np.ones(seq_len + 1)
    if full:
        fwd .fill(0.0)
        fwd[0] = 1.0
    score = 0.0

    for p in post:
        fwd, fprev = fprev, fwd

        #  Emit blank and stay in current state
        fwd = fprev * p[-1]
        #  Move from previous state and emit new character
        fwd[1:] += fprev[:-1] * p[seq]

        m = np.sum(fwd)
        fwd /= m
        score += np.log(m)

    return score + (np.log(fwd[-1]) if full else 0.0)


def backwards(post, seq):
    """ The backwards score for sequence

    :param post: A 2D :class:`ndarray`
    :param seq: Sequence to map against

    :returns: score
    """
    pass


def forwards_transposed(post, seq, skip_prob=0.0):
    """ Forwards score but computed through sequence

    Demonstrate that the forward score for a transducer can be computed by
    iterating through the sequence.  This shows the possibility of an efficient
    iterative refinement of the sequence.

    :param post: A 2D :class:`ndarray`
    :param seq: Sequence to map against
    :param skip_prob: Probability of skip

    :returns: score
    """
    nev, nstate = post.shape

    fprev = np.zeros(nev)
    fwd = np.cumprod(post[:, -1])
    m = np.sum(fwd)
    fwd /= m
    score = np.log(m)

    for s in seq:
        fwd, fprev = fprev, fwd

        # Iteration through events
        fwd = fprev * skip_prob
        fwd[1:] += fprev[:-1] * post[1:, s]
        for i in xrange(1, nev):
            fwd[i] += fwd[i - 1] * post[i, -1]

        m = np.sum(fwd)
        fwd /= m
        score += np.log(m)

    return score + np.log(fwd[-1])
