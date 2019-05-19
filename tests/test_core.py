#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

"""Tests for `mass_ts` package."""

import os

import pytest

import numpy as np

from mass_ts import core as mtscore


def test_is_array_like_invalid():
    assert(mtscore.is_array_like(1) == False)
    assert(mtscore.is_array_like('adf') == False)
    assert(mtscore.is_array_like({'a': 1}) == False)
    assert(mtscore.is_array_like(set([1, 2, 3])) == False)


def test_is_array_like_valid():
    assert(mtscore.is_array_like(np.array([1])) == True)
    assert(mtscore.is_array_like([1, ]) == True)
    assert(mtscore.is_array_like((1, 2,)) == True)


def test_is_one_dimensional_invalid():
    a = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ])
    assert(mtscore.is_one_dimensional(a) == False)


def test_is_one_dimensional_valid():
    a = np.array([1, 2, 3, 4])
    assert(mtscore.is_one_dimensional(a) == True)


def test_to_np_array_exception():
    with pytest.raises(ValueError) as excinfo:
        mtscore.to_np_array('s')
        assert 'Unable to convert to np.ndarray!' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        mtscore.to_np_array(1)
        assert 'Unable to convert to np.ndarray!' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        mtscore.to_np_array(set([1, 2, 3]))
        assert 'Unable to convert to np.ndarray!' in str(excinfo.value)


def test_to_np_array_valid():
    actual = mtscore.to_np_array([1, 2, 3])
    desired = np.array([1, 2, 3])
    np.testing.assert_equal(actual, desired)

    actual = mtscore.to_np_array((1, 2, 3,))
    desired = np.array([1, 2, 3])
    np.testing.assert_equal(actual, desired)

    actual = mtscore.to_np_array(np.array([1, 2, 3]))
    desired = np.array([1, 2, 3])
    np.testing.assert_equal(actual, desired)


def test_precheck_series_and_query_valid():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    q = [1, 2, 3, 4]

    actual_ts, actual_q = mtscore.precheck_series_and_query(ts, q)
    np.testing.assert_equal(actual_ts, np.array(ts))
    np.testing.assert_equal(actual_q, np.array(q))


def test_precheck_series_and_query_invalid():
    with pytest.raises(ValueError) as excinfo:
        mtscore.precheck_series_and_query('1', [1, 2, 3])
        assert 'Invalid ts value given. Must be array_like!' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        mtscore.precheck_series_and_query([1, 2, 3], '1')
        assert 'Invalid query value given. Must be array_like!' \
            in str(excinfo.value)


def test_rolling_window():
    a = np.array([1, 2, 3, 4, 5, 6])
    actual = mtscore.rolling_window(a, 3)
    desired = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ])

    np.testing.assert_equal(actual, desired)


def test_moving_average():
    a = np.array([1, 2, 3, 4, 5, 6])
    actual = mtscore.moving_average(a, 3)
    desired = np.array([2., 3., 4., 5.])

    np.testing.assert_equal(actual, desired)


def test_moving_std():
    a = np.array([1, 2, 3, 4, 5, 6])
    actual = mtscore.moving_std(a, 3)
    desired = np.array([0.81649658, 0.81649658, 0.81649658, 0.81649658])

    np.testing.assert_almost_equal(actual, desired)