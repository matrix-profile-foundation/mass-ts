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


def test_top_k_motifs():
    """Sanity check that compares results from UCR use case."""
    robot_dog = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'robot_dog.txt'))
    carpet_walk = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'carpet_query.txt'))

    distances = mts.mass2(robot_dog, carpet_walk)
    found = mts.top_k_motifs(distances, 2, 25)
    found = np.array(found)
    expected = np.array([7479, 6999])

    assert(np.array_equal(found, expected))


def test_top_k_discords():
    """Sanity check that compares results from UCR use case."""
    robot_dog = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'robot_dog.txt'))
    carpet_walk = np.loadtxt(
        os.path.join(MODULE_PATH, '..', 'tests', 'carpet_query.txt'))

    distances = mts.mass2(robot_dog, carpet_walk)
    found = mts.top_k_discords(distances, 2, 25)
    found = np.array(found)
    expected = np.array([12900, 2])

    assert(np.array_equal(found, expected))