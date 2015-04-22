simplebayes
===========
A memory-based, optional-persistence naïve bayesian text classifier.
--------------------------------------------------------------------
::

    This work is heavily inspired by the python "redisbayes" module found here:
    [https://github.com/jart/redisbayes] and [https://pypi.python.org/pypi/redisbayes]

    I've elected to write this to alleviate the network/time requirements when
    using the bayesian classifier to classify large sets of text, or when
    attempting to train with very large sets of sample data.

Build Status
------------
.. image:: https://travis-ci.org/hickeroar/simplebayes.svg?branch=master
.. image:: https://img.shields.io/badge/coverage-100%-brightgreen.svg?style=flat
.. image:: https://img.shields.io/badge/pylint-10.00/10-brightgreen.svg?style=flat
.. image:: https://img.shields.io/badge/flake8-passing-brightgreen.svg?style=flat

Installation
------------
::

    sudo pip install simplebayes

Basic Usage
-----------
.. code-block:: python

    import simplebayes
    bayes = simplebayes.SimpleBayes()

    bayes.train('good', 'sunshine drugs love sex lobster sloth')
    bayes.train('bad', 'fear death horror government zombie')

    assert bayes.classify('sloths are so cute i love them') == 'good'
    assert bayes.classify('i would fear a zombie and love the government') == 'bad'

    print bayes.score('i fear zombies and love the government')

    bayes.untrain('bad', 'fear death')

    assert bayes.tally('bad') == 3

Cache Usage
-----------
.. code-block:: python

    import simplebayes
    bayes = simplebayes.SimpleBayes(cache_path='/my/cache/')
    # Cache file is '/my/cache/_simplebayes.pickle'
    # Default cache_path is '/tmp/'

    if not bayes.cache_train():
        # Unable to load cache data, so we're training it
        bayes.train('good', 'sunshine drugs love sex lobster sloth')
        bayes.train('bad', 'fear death horror government zombie')

        # Saving the cache so next time the training won't be needed
        bayes.persist_cache()

Tokenizer Override
------------------
.. code-block:: python

    import simplebayes

    def my_tokenizer(sample):
        return sample.split()

    bayes = simplebayes.SimpleBayes(tokenizer=my_tokenizer)

License
-------
::

    The MIT License (MIT)

    Copyright (c) 2015 Ryan Vennell

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

API Documentation
-----------------
`<http://hickeroar.github.io/simplebayes/simplebayes.html>`_