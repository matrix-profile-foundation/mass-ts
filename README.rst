====
MASS
====


.. image:: https://img.shields.io/pypi/v/mass_ts.svg
        :target: https://pypi.python.org/pypi/mass_ts

.. image:: https://img.shields.io/travis/tylerwmarrs/mass_ts.svg
        :target: https://travis-ci.org/tylerwmarrs/mass_ts

.. image:: https://readthedocs.org/projects/mass-ts/badge/?version=latest
        :target: https://mass-ts.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




MASS (Mueen's Algorithm for Similarity Search) Implementations


* Free software: Apache Software License 2.0


Features
--------

* MASS
* MASS2
* MASS3 - TODO

Example Usage
-------------
.. code:: python
    import numpy as np
    import mass_ts

    ts = np.loadtxt('ts.txt')
    query = np.loadtxt('query.txt')

    distances = mass_ts.mass2(ts, query)

Citations
---------
Abdullah Mueen, Yan Zhu, Michael Yeh, Kaveh Kamgar, Krishnamurthy Viswanathan, Chetan Kumar Gupta and Eamonn Keogh (2015), The Fastest Similarity Search Algorithm for Time Series Subsequences under Euclidean Distance, URL: http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
