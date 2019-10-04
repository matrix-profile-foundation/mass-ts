# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import warnings

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    warnings.warn(
        'GPU support will not work. You must pip install mass-ts[gpu].')

from mass_ts import core as mtscore


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
    ts, query = mtscore.precheck_series_and_query(ts, query)

    if normalize_query:
        query = (query - np.mean(query)) / np.std(query)
        
    n = len(ts)
    m = len(query)
    x = np.append(ts, np.zeros([1, n]))
    y = np.append(np.flipud(query), np.zeros([1, 2 * n - m]))
    
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    Y.resize(X.shape)
    Z = X * Y
    z = np.fft.ifft(Z)
    
    sumy = np.sum(y)
    sumy2 = np.sum(y ** 2)
    
    cum_sumx = np.cumsum(x)
    cum_sumx2 = np.cumsum(x ** 2)
    
    sumx2 = cum_sumx2[m:n] - cum_sumx2[0:n-m]
    sumx = cum_sumx[m:n] - cum_sumx[0:n-m]
    meanx = sumx / m
    sigmax2 = (sumx2 / m) - (meanx**2)
    sigmax = np.sqrt(sigmax2)
    
    dist = (sumx2 - 2 * sumx * meanx + m * (meanx ** 2)) \
        / sigmax2 - 2 * (z[m:n] - sumy * meanx) \
        / sigmax + sumy2
    dist = np.absolute(np.sqrt(dist))
    
    if corr_coef:
        return 1 - np.absolute(dist) / (2 * m)
    
    return dist


def mass2_gpu(ts, query):
    """
    Compute the distance profile for the given query over the given time 
    series. This require cupy to be installed.

    Parameters
    ----------
    ts : array_like
        The array to create a rolling window on.
    query : array_like
        The query.

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
    def moving_mean_std_gpu(a, w):
        s = cp.concatenate([cp.array([0]), cp.cumsum(a)])
        sSq = cp.concatenate([cp.array([0]), cp.cumsum(a ** 2)])
        segSum = s[w:] - s[:-w]
        segSumSq = sSq[w:] -sSq[:-w]
    
        movmean = segSum / w
        movstd = cp.sqrt(segSumSq / w - (segSum / w) ** 2)
    
        return (movmean, movstd)

    x = cp.asarray(ts)
    y = cp.asarray(query)
    n = x.size
    m = y.size

    meany = cp.mean(y)
    sigmay = cp.std(y)
    
    meanx, sigmax = moving_mean_std_gpu(x, m)
    meanx = cp.concatenate([cp.ones(n - meanx.size), meanx])
    sigmax = cp.concatenate([cp.zeros(n - sigmax.size), sigmax])
    
    y = cp.concatenate((cp.flip(y, axis=0), cp.zeros(n - m)))
    
    X = cp.fft.fft(x)
    Y = cp.fft.fft(y)
    Z = X * Y
    z = cp.fft.ifft(Z)
    
    dist = 2 * (m - (z[m - 1:n] - m * meanx[m - 1:n] * meany) / 
                    (sigmax[m - 1:n] * sigmay))
    dist = cp.sqrt(dist)

    return cp.asnumpy(dist)


def mass2(ts, query):
    """
    Compute the distance profile for the given query over the given time 
    series. Optionally, the correlation coefficient can be returned.

    Parameters
    ----------
    ts : array_like
        The array to create a rolling window on.
    query : array_like
        The query.

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
    ts, query = mtscore.precheck_series_and_query(ts, query)

    n = len(ts)
    m = len(query)
    x = ts
    y = query

    meany = np.mean(y)
    sigmay = np.std(y)
    
    meanx = mtscore.moving_average(x, m)
    meanx = np.append(np.ones([1, len(x) - len(meanx)]), meanx)
    sigmax = mtscore.moving_std(x, m)
    sigmax = np.append(np.zeros([1, len(x) - len(sigmax)]), sigmax)
    
    y = np.append(np.flip(y), np.zeros([1, n - m]))
    
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    Y.resize(X.shape)
    Z = X * Y
    z = np.fft.ifft(Z)
    
    dist = 2 * (m - (z[m - 1:n] - m * meanx[m - 1:n] * meany) / 
                    (sigmax[m - 1:n] * sigmay))
    dist = np.sqrt(dist)
    
    return dist


def mass3(ts, query, pieces):
    """
    Compute the distance profile for the given query over the given time 
    series. This version of MASS is hardware efficient given the right number
    of pieces.

    Parameters
    ----------
    ts : array_like
        The array to create a rolling window on.
    query : array_like
        The query.
    pieces : int
        Number of pieces to process. This is best as a power of 2.

    Returns
    -------
    An array of distances.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
        If pieces is less than the length of the query.
    """
    ts, query = mtscore.precheck_series_and_query(ts, query)

    m = len(query)
    
    if pieces < m:
        raise ValueError('pieces should be larger than the query length.')
    
    n = len(ts)
    k = pieces
    x = ts
    dist = np.array([])
    
    # compute stats in O(n)
    meany = np.mean(query)
    sigmay = np.std(query)
    
    meanx = mtscore.moving_average(x, m)
    meanx = np.append(np.ones([1, len(x) - len(meanx)]), meanx)
    sigmax = mtscore.moving_std(x, m)
    sigmax = np.append(np.zeros([1, len(x) - len(sigmax)]), sigmax)
    
    # reverse the query and append zeros
    y = np.append(np.flip(query), np.zeros(pieces - m))
    
    step_size = k - m + 1
    stop = n - k + 1
       
    for j in range(0, stop, step_size):
        # The main trick of getting dot products in O(n log n) time
        X = np.fft.fft(x[j:j + k])
        Y = np.fft.fft(y)
        
        Z = X * Y
        z = np.fft.ifft(Z)
            
        d = 2 * (m-(z[m - 1:k] - m * meanx[m + j - 1:j + k] * meany) /
                   (sigmax[m + j - 1:j + k] * sigmay))
        d = np.sqrt(d)
        dist = np.append(dist, d)
   
    j = j + k - m
    k = n - j - 1
    if k >= m:
        X = np.fft.fft(x[j:n-1])
        y = y[0:k]

        Y = np.fft.fft(y)
        Z = X * Y
        z = np.fft.ifft(Z)

        d = 2 * (m-(z[m - 1:k] - m * meanx[j + m - 1:n - 1] * meany) /
                 (sigmax[j + m - 1:n - 1] * sigmay))
       
        d = np.sqrt(d)
        dist = np.append(dist, d)
    
    return np.array(dist)