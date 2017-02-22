from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
NBASE = 4


def nkmer(kmer, base=NBASE):
    """  Number of possible kmers of a given length

    :param kmer: Length of kmer
    :param base: Number of letters in alphabet

    :returns: Number of kmers
    """
    return base ** kmer


def nstate(kmer, transducer=True, bad_state=True, base=NBASE):
    """  Number of states in model

    :param kmer: Length of kmer
    :param transducer: Is the model a transducer?
    :param bad_state: Does the model have a bad state
    :param base: Number of letters in alphabet

    :returns: Number of states
    """
    return nkmer(kmer, base=base) + (transducer or bad_state)
