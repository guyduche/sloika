from functools import partial
from scipy.stats import truncnorm
from sloika import sloika_dtype
from sloika.activation import *
from sloika.layers import *

_NBASE = 4

def _rn(size, sd):
    #  Truncated normal for Xavier style initiation
    res = sd * truncnorm.rvs(-2, 2, size=size)
    return res.astype(sloika_dtype)
