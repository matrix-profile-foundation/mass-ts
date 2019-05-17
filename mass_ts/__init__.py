# -*- coding: utf-8 -*-

"""
MASS (Mueen's Algorithm for Similarity Search)
==============================================

Provides
	1. MASS - the original implementation
	2. MASS2 - a quicker version of MASS
	3. MASS3 - a piecewise version of MASS2

Example Usage
-------------
>>> import mass_ts
>>> import numpy as np
>>> ts = np.loadtxt('ts.txt')
>>> query = np.loadtxt('query.txt')
>>> distance = mass_ts.mass(ts, query)

Citations
---------
Abdullah Mueen, Yan Zhu, Michael Yeh, Kaveh Kamgar, Krishnamurthy Viswanathan, Chetan Kumar Gupta and Eamonn Keogh (2015), The Fastest Similarity Search Algorithm for Time Series Subsequences under Euclidean Distance, URL: http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
"""

__author__ = """Tyler Marrs"""
__email__ = 'tylerwmarrs@gmail.com'
__version__ = '0.1.1'


from mass_ts._mass_ts import mass, mass2, mass3