# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np


def _is_array_like(a):
    """
    Helper function to determine if a value is array like.

    a : obj
        Object to test.

    Returns
    -------
    True or false respectively.
    """
    return isinstance(a, (list, tuple, np.ndarray))


def _is_one_dimensional(a):
    """
    Helper function to determine if value is one dimensional.

    a : array_like
        Object to test.

    Returns
    -------
    True or false respectively.
    """
    return a.ndim == 1


def _to_np_array(a):
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
    if not _is_array_like(a):
        raise ValueError('Unable to convert to np.ndarray!')

    return np.array(a)


def _rolling_window(a, window):
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


def _moving_average(a, window=3):
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
    return np.mean(_rolling_window(a, window), -1)


def _moving_std(a, window=3):
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
    return np.std(_rolling_window(a, window), -1)


def _precheck_series_and_query(ts, query):
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
        ts = _to_np_array(ts)
    except ValueError:
        raise ValueError('Invalid ts value given. Must be array_like!')

    try:
        query = _to_np_array(query)
    except ValueError:
        raise ValueError('Invalid query value given. Must be array_like!')


    if not _is_one_dimensional(ts):
        raise ValueError('ts must be one dimensional!')

    if not _is_one_dimensional(query):
        raise ValueError('query must be one dimensional!')

    return (ts, query)


def mass(ts, query, normalize_query=True, corr_coef=False):
    """
    Compute the distance profile for the given query over the given time 
    series. Optionally, the correlation coefficient can be returned and 
    the query can be normalized.

    Parameters
    ----------
    ts : array_like
        The array to create a rolling window on.
    query : array_like
        The query.
    normalize_query : bool, default True
        Optionally normalize the query.
    corr_coef : bool, default False
        Optionally return the correlation coef.

    Returns
    -------
    An array of distances.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
    """
    ts, query = _precheck_series_and_query(ts, query)

    if normalize_query:
        query = (query - np.mean(query)) / np.std(query)
        
    n = len(ts)
    m = len(query)
    x = np.append(ts, np.zeros([1, n]))
    query = np.append(np.flipud(query), np.zeros([1, m]))
    
    X = np.fft.fft(ts)
    Y = np.fft.fft(query)
    Y.resize(X.shape)
    Z = X * Y
    z = np.fft.ifft(Z)
    
    sum_query = np.sum(query)
    sum_query2 = np.sum(query**2)
    
    cum_sumx = np.cumsum(x)
    cum_sumx2 = np.cumsum(x**2)
    
    sumx2 = cum_sumx2[m:n] - cum_sumx2[0:n-m]
    sumx = cum_sumx[m:n] - cum_sumx[0:n-m]
    meanx = sumx / m
    sigmax2 = (sumx2 / m) - (meanx**2)
    sigmax = np.sqrt(sigmax2)
    
    dist = (sumx2 - 2 * sumx * meanx + m * (meanx ** 2)) \
        / sigmax2 - 2 * (z[m:n] - sum_query * meanx) \
        / sigmax + sum_query2
    dist = np.absolute(np.sqrt(dist))
    
    if corr_coef:
        return 1 - np.absolute(dist) / (2 * m)
    
    return dist


def mass2(ts, query, corr_coef=False):
    """
    Compute the distance profile for the given query over the given time 
    series. Optionally, the correlation coefficient can be returned.

    Parameters
    ----------
    ts : array_like
        The array to create a rolling window on.
    query : array_like
        The query.
    corr_coef : bool, default False
        Optionally return the correlation coef.

    Returns
    -------
    An array of distances.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
    """
    ts, query = _precheck_series_and_query(ts, query)

    n = len(ts)
    m = len(query)
    ts = np.append(ts, np.zeros([1, n]))
    
    meanquery = np.mean(query)
    sigmaquery = np.std(query)
    
    meanx = _moving_average(ts, m-1)
    sigmax = _moving_std(ts, m-1)
    
    query = np.append(np.flipud(query), np.zeros([1, m]))
    
    X = np.fft.fft(ts)
    Y = np.fft.fft(query)
    Y.resize(X.shape)
    Z = X * Y
    z = np.fft.ifft(Z)
    
    dist = 2 * (m - (z[m:n] - m * meanx[m:n] * meanquery) / (sigmax[m:n] * sigmaquery));
    
    if corr_coef:
        return 1 - np.absolute(dist) / (2 * m)
    
    return dist


def mass3():
    raise NotImplementedError('To be implemented')