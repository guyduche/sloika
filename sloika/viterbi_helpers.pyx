import  numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ITYPE = np.int
ctypedef np.int_t ITYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def slip_update(np.ndarray[DTYPE_t, ndim=1] x, DTYPE_t slip):
    cdef int j
    cdef np.ndarray[DTYPE_t, ndim=1] from_score = np.zeros(len(x), dtype=DTYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] from_pos = np.zeros(len(x), dtype=ITYPE)

    from_score[0] = from_score[1] = -1e38
    from_score[2] = x[0] - slip
    from_pos[2] = 0
    for j in range(3, len(x)):
        if from_score[j - 1] >= x[j - 2]:
            from_pos[j] = from_pos[j - 1]
            from_score[j] = from_score[j - 1]
        else:
            from_pos[j] = j - 2
            from_score[j] = x[j - 2]
        from_score[j] -= slip
    
    return from_score, from_pos
