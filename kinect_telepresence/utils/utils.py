import numpy as np

def check_mtrx(mtrx, what_mtrx = "", required_shape = (3, 3)):
    '''
    Checker wether matrix has required shape or not
    '''
    if type(mtrx) != np.ndarray:
        raise TypeError(f"Wrong matrix type. {what_mtrx} is not np.ndarray")
    if mtrx.shape != required_shape:
        raise ValueError(f"Wrong matrix. {what_mtrx} should have {required_shape} shape, but has {mtrx.shape}")

