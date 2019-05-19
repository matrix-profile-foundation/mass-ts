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

import mass_ts as mts

MODULE_PATH = mts.__path__[0]


def test_mass2_batch_robotdog_single_threaded():
    """Sanity check that compares results from UCR use case."""
    robot_dog = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'robot_dog.txt'))
    carpet_walk = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'carpet_query.txt'))

    indices, distances = mts.mass2_batch(
        robot_dog, carpet_walk, 1000, top_matches=3)
    min_dist_idx = np.argmin(distances)
    min_idx = indices[min_dist_idx]

    assert(min_idx == 7479)


def test_mass2_batch_robotdog_multi_threaded():
    """Sanity check that compares results from UCR use case."""
    robot_dog = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'robot_dog.txt'))
    carpet_walk = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'carpet_query.txt'))

    indices, distances = mts.mass2_batch(
        robot_dog, carpet_walk, 1000, top_matches=3, n_jobs=2)
    min_dist_idx = np.argmin(distances)
    min_idx = indices[min_dist_idx]

    assert(min_idx == 7479)