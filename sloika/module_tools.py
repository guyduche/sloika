from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
from functools import partial
from scipy.stats import truncnorm
from sloika.config import sloika_dtype
from sloika.activation import *
from sloika.layers import *
from sloika.variables import *


def _rn(size, sd):
    #  Truncated normal for Xavier style initiation
    res = sd * truncnorm.rvs(-2, 2, size=size)
    return res.astype(sloika_dtype)
