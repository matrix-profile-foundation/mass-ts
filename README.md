MASS
----

[<img src="https://img.shields.io/pypi/v/mass_ts.svg">](https://pypi.python.org/pypi/mass_ts)
[<img src="https://img.shields.io/travis/tylerwmarrs/mass_ts.svg">](https://travis-ci.org/tylerwmarrs/mass_ts)
[<img src="https://readthedocs.org/projects/mass-ts/badge/?version=latest">](https://mass-ts.readthedocs.io/en/latest/?badge=latest)


MASS (Mueen's Algorithm for Similarity Search) Implementations


* Free software: Apache Software License 2.0


Features
--------

* MASS
* MASS2
* MASS3 - TODO

Example Usage
-------------
```python

ts = np.loadtxt('ts.txt')
query = np.loadtxt('query.txt')

distances = mass_ts.mass2(ts, query)
```

Citations
---------
Abdullah Mueen, Yan Zhu, Michael Yeh, Kaveh Kamgar, Krishnamurthy Viswanathan, Chetan Kumar Gupta and Eamonn Keogh (2015), The Fastest Similarity Search Algorithm for Time Series Subsequences under Euclidean Distance, URL: http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
