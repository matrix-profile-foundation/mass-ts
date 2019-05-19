# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import multiprocessing
import sys

import numpy as np


def mp_pool():
    """
    Utility function to get the appropriate multiprocessing
    handler for Python 2 and 3.
    """
    ctxt = None
    if sys.version_info[0] == 2:
        from contextlib import contextmanager

        @contextmanager
        def multiprocessing_context(*args, **kwargs):
            pool = multiprocessing.Pool(*args, **kwargs)
            yield pool
            pool.terminate()

        ctxt = multiprocessing_context
    else:
        ctxt = multiprocessing.Pool

    return ctxt


def is_array_like(a):
    """
    Helper function to determine if a value is array like.

    a : obj
        Object to test.

    Returns
    -------
    True or false respectively.
    """
    return isinstance(a, (list, tuple, np.ndarray))


def is_one_dimensional(a):
    """
    Helper function to determine if value is one dimensional.

    a : array_like
        Object to test.

    Returns
    -------
    True or false respectively.
    """
    return a.ndim == 1


def to_np_array(a):
    """
    Helper function to convert tuple or list to np.ndarray.

    a : Tuple, list or np.ndarray
        The object to transform.

    Returns
    -------
    The np.ndarray.

    Raises
    ------
    ValueError
        If a is not a valid type.
    """
    if not is_array_like(a):
        raise ValueError('Unable to convert to np.ndarray!')

    return np.array(a)


def rolling_window(a, window):
    """
    Provides a rolling window on a numpy array given an array and window size.

    Parameters
    ----------
    a : array_like
        The array to create a rolling window on.
    window : int
        The window size.

    Returns
    -------
    Strided array for computation.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def moving_average(a, window=3):
    """
    Computes the moving average over an array given a window size.

    Parameters
    ----------
    a : array_like
        The array to compute the moving average on.
    window : int
        The window size.

    Returns
    -------
    The moving average over the array.
    """
    return np.mean(rolling_window(a, window), -1)


def moving_std(a, window=3):
    """
    Computes the moving std. over an array given a window size.

    Parameters
    ----------
    a : array_like
        The array to compute the moving std. on.
    window : int
        The window size.

    Returns
    -------
    The moving std. over the array.
    """
    return np.std(rolling_window(a, window), -1)


def precheck_series_and_query(ts, query):
    """
    Helper function to ensure we have 1d time series and query.

    Parameters
    ----------
    ts : array_like
        The array to create a rolling window on.
    query : array_like
        The query.

    Returns
    -------
    (np.array, np.array) - The ts and query respectively.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
    """
    try:
        ts = to_np_array(ts)
    except ValueError:
        raise ValueError('Invalid ts value given. Must be array_like!')

    try:
        query = to_np_array(query)
    except ValueError:
        raise ValueError('Invalid query value given. Must be array_like!')

    if not is_one_dimensional(ts):
        raise ValueError('ts must be one dimensional!')

    if not is_one_dimensional(query):
        raise ValueError('query must be one dimensional!')

    return (ts, query)