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

import numpy as np

from mass_ts import core as mtscore


def _top_k(distance_profile, k, exclusion_zone, option):
    """
    Finds top k discords or motifs given an exclusion zone. The exclusion zone
    acts as a buffer between a found index on the left and right hand side. 
    For example, if you set the exclusion zone to 4 and a motif or discord was 
    found at index 100, the algorithm ignores indices 96 through 104 on 
    subsequent iterations.
    
    Parameters
    ----------
    distance_profile : array_like
        The distance profile computed by a MASS algorithm.
    k : int
        The number of results you want returned.
    exclusion_zone : int
        The buffer around a found index to exclude results from being apart of.
    option : str ('motifs', 'discords')
        Specify if you want to find motifs or discords.
    
    Returns
    -------
    Top k discord or motif starting indices as a list of integers.
    
    Raises
    ------
    ValueError
        If distance_profile is not array_like.
        If k is not an integer or is less than 1.
        If option is not discords or motifs.
        If exclusion_zone is not an integer or is less than 1.
    """
    # perform value checking
    if not mtscore.is_array_like(distance_profile):
        raise ValueError('distance_profile must be array like.')
    
    # check k
    if not isinstance(k, int) or k < 1:
        raise ValueError('k must be an integer of 1 or more.')
    
    # check for discords or motifs as option
    option = option.lower()
    if option not in ('discords', 'motifs'):
        raise ValueError('option only accepts discords or motifs.')
    
    if not isinstance(exclusion_zone, int) or exclusion_zone < 1:
        raise ValueError('exclusion_zone must be an integer of 1 or more.')

    distance_profile = mtscore.to_np_array(distance_profile)
    n = len(distance_profile)
    found = []
    tmp = distance_profile.copy()
    
    # obtain indices in ascending order
    indices = np.argpartition(tmp, k)
    
    # created flipped view for discords
    if option == 'discords':
        indices = indices[::-1]

    for idx in indices:
        if not np.isinf(tmp[idx]):
            found.append(idx)

        # apply exclusion zone
        exclusion_zone_start = np.max([0, idx - exclusion_zone])
        exclusion_zone_end = np.min([n, idx + exclusion_zone])
        tmp[exclusion_zone_start:exclusion_zone_end] = np.inf

        if len(found) >= k:
            break

    return found


def top_k_motifs(distance_profile, k, exclusion_zone):
    """
    Finds top k motifs given an exclusion zone. The exclusion zone acts as a 
    buffer between a found index on the left and right hand side. For example, 
    if you set the exclusion zone to 4 and a motif was found at 
    index 100, the algorithm ignores indices 96 through 104 on  subsequent 
    iterations.
    
    Parameters
    ----------
    distance_profile : array_like
        The distance profile computed by a MASS algorithm.
    k : int
        The number of results you want returned.
    exclusion_zone : int
        The buffer around a found index to exclude results from being apart of.
    
    Returns
    -------
    Top k motif starting indices as a list of integers.
    
    Raises
    ------
    ValueError
        If distance_profile is not array_like.
        If k is not an integer or is less than 1.
        If exclusion_zone is not an integer or is less than 1.
    """
    return _top_k(distance_profile, k, exclusion_zone, 'motifs')


def top_k_discords(distance_profile, k, exclusion_zone):
    """
    Finds top k discords given an exclusion zone. The exclusion zone acts as a 
    buffer between a found index on the left and right hand side. For example, 
    if you set the exclusion zone to 4 and a discord was found at 
    index 100, the algorithm ignores indices 96 through 104 on  subsequent 
    iterations.
    
    Parameters
    ----------
    distance_profile : array_like
        The distance profile computed by a MASS algorithm.
    k : int
        The number of results you want returned.
    exclusion_zone : int
        The buffer around a found index to exclude results from being apart of.
    
    Returns
    -------
    Top k motif starting indices as a list of integers.
    
    Raises
    ------
    ValueError
        If distance_profile is not array_like.
        If k is not an integer or is less than 1.
        If exclusion_zone is not an integer or is less than 1.
    """
    return _top_k(distance_profile, k, exclusion_zone, 'discords')