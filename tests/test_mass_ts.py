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

import mass_ts
from mass_ts import _mass_ts as mts

MODULE_PATH = mass_ts.__path__[0]


def test_is_array_like_invalid():
    assert(mts._is_array_like(1) == False)
    assert(mts._is_array_like('adf') == False)
    assert(mts._is_array_like({'a': 1}) == False)
    assert(mts._is_array_like(set([1, 2, 3])) == False)


def test_is_array_like_valid():
    assert(mts._is_array_like(np.array([1])) == True)
    assert(mts._is_array_like([1, ]) == True)
    assert(mts._is_array_like((1, 2,)) == True)


def test_is_one_dimensional_invalid():
    a = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ])
    assert(mts._is_one_dimensional(a) == False)


def test_is_one_dimensional_valid():
    a = np.array([1, 2, 3, 4])
    assert(mts._is_one_dimensional(a) == True)


def test_to_np_array_exception():
    with pytest.raises(ValueError) as excinfo:
        mts._to_np_array('s')
        assert 'Unable to convert to np.ndarray!' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        mts._to_np_array(1)
        assert 'Unable to convert to np.ndarray!' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        mts._to_np_array(set([1, 2, 3]))
        assert 'Unable to convert to np.ndarray!' in str(excinfo.value)


def test_to_np_array_valid():
    actual = mts._to_np_array([1, 2, 3])
    desired = np.array([1, 2, 3])
    np.testing.assert_equal(actual, desired)

    actual = mts._to_np_array((1, 2, 3,))
    desired = np.array([1, 2, 3])
    np.testing.assert_equal(actual, desired)

    actual = mts._to_np_array(np.array([1, 2, 3]))
    desired = np.array([1, 2, 3])
    np.testing.assert_equal(actual, desired)


def test_precheck_series_and_query_valid():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    q = [1, 2, 3, 4]

    actual_ts, actual_q = mts._precheck_series_and_query(ts, q)
    np.testing.assert_equal(actual_ts, np.array(ts))
    np.testing.assert_equal(actual_q, np.array(q))


def test_precheck_series_and_query_invalid():
    with pytest.raises(ValueError) as excinfo:
        mts._precheck_series_and_query('1', [1, 2, 3])
        assert 'Invalid ts value given. Must be array_like!' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        mts._precheck_series_and_query([1, 2, 3], '1')
        assert 'Invalid query value given. Must be array_like!' \
            in str(excinfo.value)


def test_rolling_window():
    a = np.array([1, 2, 3, 4, 5, 6])
    actual = mts._rolling_window(a, 3)
    desired = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ])

    np.testing.assert_equal(actual, desired)


def test_moving_average():
    a = np.array([1, 2, 3, 4, 5, 6])
    actual = mts._moving_average(a, 3)
    desired = np.array([2., 3., 4., 5.])

    np.testing.assert_equal(actual, desired)


def test_moving_std():
    a = np.array([1, 2, 3, 4, 5, 6])
    actual = mts._moving_std(a, 3)
    desired = np.array([0.81649658, 0.81649658, 0.81649658, 0.81649658])

    np.testing.assert_almost_equal(actual, desired)


def test_mass():
    ts = np.array([1, 1, 1, 2, 1, 1, 4, 5])
    query = np.array([2, 1, 1, 4])
    actual = mts.mass(ts, query)
    desired = np.array([
        3.43092352e+00, 3.43092352e+00, 2.98023224e-08, 1.85113597e+00
    ])
    
    np.testing.assert_almost_equal(actual, desired)


def test_mass_corr_coef():
    ts = np.array([1, 1, 1, 2, 1, 1, 4, 5])
    query = np.array([2, 1, 1, 4])
    actual = mts.mass(ts, query, corr_coef=True)
    desired = np.array([0.57113456, 0.57113456, 1., 0.768608])
    
    np.testing.assert_almost_equal(actual, desired)


def test_mass2():
    ts = np.array([1, 1, 1, 2, 1, 1, 4, 5])
    query = np.array([2, 1, 1, 4])
    actual = mts.mass2(ts, query)
    desired = np.array([
        0.67640791-1.37044402e-16j,
        3.43092352+0.00000000e+00j,
        3.43092352+1.02889035e-17j,
        0.+0.00000000e+00j,
        1.85113597+1.21452707e-17j
    ])
    
    np.testing.assert_almost_equal(actual, desired)


def test_mass3():
    ts = np.array([1, 1, 1, 2, 1, 1, 4, 5])
    query = np.array([2, 1, 1, 4])
    pieces = 8
    distances = mts.mass3(ts, query, pieces)
    desired = np.array([
        0.67640791,
        3.43092352,
        3.43092352,
        0.,
        1.85113597
    ])

    np.testing.assert_almost_equal(distances, desired)


def test_mass_robotdog():
    """Sanity check that compares results from UCR use case."""
    robot_dog = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'robot_dog.txt'))
    carpet_walk = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'carpet_query.txt'))

    distances = mts.mass(robot_dog, carpet_walk)
    min_idx = np.argmin(distances) + 1

    assert(min_idx == 7479)


def test_mass2_robotdog():
    """Sanity check that compares results from UCR use case."""
    robot_dog = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'robot_dog.txt'))
    carpet_walk = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'carpet_query.txt'))

    distances = mts.mass2(robot_dog, carpet_walk)
    min_idx = np.argmin(distances)

    assert(min_idx == 7479)


def test_mass3_robotdog():
    """Sanity check that compares results from UCR use case."""
    robot_dog = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'robot_dog.txt'))
    carpet_walk = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'carpet_query.txt'))

    distances = mts.mass3(robot_dog, carpet_walk, 256)
    min_idx = np.argmin(distances)

    assert(min_idx == 7479)