# from http://www.johnvinyard.com/blog/?p=268
# Refer to https://github.com/sussexwearlab/DeepConvLSTM for details of this module
# and for citation

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


def norm_shape(shape):
    """
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    """
    try:
        i = int(shape)
        return i,
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    """
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    """

    if ss is None:
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)


    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError('a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError('ws cannot be larger than a in any dimension.\
     a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    # how many slices will there be in each dimension?

    new_shape = norm_shape(((shape - ws) // ss) + 1)

    new_shape += norm_shape(ws)

    new_strides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=new_shape, strides=new_strides)
    if not flatten:
        return strided

    meat = len(ws) if ws.shape else 0
    first_dim = (np.product(new_shape[:-meat]),) if ws.shape else ()
    dim = first_dim + (new_shape[-meat:])
    # remove any dimensions with size 1
    # dim = filter(lambda i: i != 1, dim)
    return strided.reshape(dim)
