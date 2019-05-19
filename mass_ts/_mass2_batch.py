# -*- coding: utf-8 -*-

"""
This module contains all logic used for the MASS2 parallel
and batch implementations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate


from multiprocessing import cpu_count

import numpy as np

import mass_ts as mts
from mass_ts import core as mtscore


def _min_subsequence_distance(values):
    """
    Computes the minimum distance for a given subsequence. The values
    consist of the iteration, batch size, subsequence and query. It is
    used for both batch processing in single threaded or multi-processing
    mode.

    Parameters
    ----------
    values : tuple(iteration, batch_size, subsequence, query)
        Tuple packed values for parallelization.

    Returns
    -------
    A tuple of the minimum index and distance for this particular subsequence.
    """
    iteration, batch_size, subsequence, query = values
    distances = mts.mass2(subsequence, query)

    # find mininimum index of this batch which will be between 0 and batch_size
    min_idx = np.argmin(distances)

    # add this distance to best distances
    dist = distances[min_idx]

    # compute the actual index and store it
    index = min_idx + (batch_size * iteration)

    return (index, dist)


def _batch_job_generator(ts, query, indices, batch_size):
    """
    A generator that yields the iteration, batch_size, subsequence
    and query for both single and multi processing.

    Parameters
    ----------
    ts : np.array
        The time series to compute similarity distances for.
    query : np.array
        The query to find matches for within the time series.
    indices : list(int)
        Indices to generate tasks for. It is primarily used for subsequence
        partitioning.
    batch_size : int
        The subsequence size.

    Returns
    -------
    A yielded job to compute.
    """
    for iteration, i in enumerate(indices):
        subsequence = ts[i:i+batch_size]

        yield (iteration, batch_size, subsequence, query)


def mass2_batch(ts, query, batch_size, top_matches=3, n_jobs=1):
    """
    MASS2 batch is a batch version of MASS2 that reduces overall memory usage,
    provides parallelization and enables you to find top K number of matches
    within the time series. The goal of using this implementation is for very
    large time series similarity search. The returned results are not sorted
    by distance. So you will need to find the top match with np.argmin() or
    sort them yourself.

    Parameters
    ----------
    ts : array_like
        The time series.
    query : array_like
        The query to search for.
    batch_size : int
        The partitioning size of the time series into batches. For example,
        a time series of length 1,000 and a batch size of 100 would create
        10 jobs where the first subsequence is 0 to 100.
    top_matches : int, Default 3
        The number of matches you would like to return.
    n_jobs : int, Default 1
        By default the implementation runs in single-threaded mode. Setting the
        n_jobs to < 1 sets the n_jobs to the number of available threads on the
        computer it is ran.

    Note
    ----
    This implementation does not support returning the entire distance profile
    at this time. However, it will be implemented in the near future.

    The value selected for top matches should be at max (n / batch_size).

    Returns
    -------
    Tuple (indices, distances) - a tuple of np.arrays where the first index is
    the indices and the second is the distances.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
        If batch_size is not an integer.
        If top_matches is < 1 or is not an integer.
        If n_jobs is not an integer.
    """
    # parameter validation
    ts, query = mtscore.precheck_series_and_query(ts, query)

    if not isinstance(batch_size, int):
        raise ValueError('batch_size must be an integer.')

    if not isinstance(top_matches, int) or top_matches < 1:
        raise ValueError('top_matches must be an integer > 0.')

    if not isinstance(n_jobs, int):
        raise ValueError('n_jobs must be an integer.')

    # set the n_jobs appropriately
    if n_jobs < 1:
        n_jobs = cpu_count()

    if n_jobs > cpu_count():
        n_jobs = cpu_count()

    n = len(ts)
    matches = []    
    
    # generate indices to process over given batch size
    indices = list(range(0, n - batch_size + 1, batch_size))
    
    # determine if we are multiprocessing or not based on cpu_count
    if n_jobs > 1:
        with mtscore.mp_pool()(processes=n_jobs) as pool:
            matches = pool.map(
                _min_subsequence_distance,
                _batch_job_generator(ts, query, indices, batch_size)
            )
    else:
        for values in _batch_job_generator(ts, query, indices, batch_size):
            matches.append(_min_subsequence_distance(values))
    
    # grab the indices and distances
    matches = np.array(matches)
    
    # find the best K number of matches
    # distance is in column 1
    top_indices = np.argpartition(matches[:, 1], top_matches)[0:top_matches]
    
    # ignore the warning when casting the index values back to ints
    # to store all values it had to choose complex types to handle the
    # distances
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings(
            'ignore',
            r'Casting complex values to real discards the imaginary part'
        )
        best_indices = matches[:, 0][top_indices].astype('int64')

    best_dists = matches[:, 1][top_indices]
    
    return (best_indices, best_dists)