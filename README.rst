simplebayes
===========
A memory-based, non-persistent naïve bayesian text classifier.
--------------------------------------------------------------

::

    This work is heavily inspired by the python "redisbayes" module found here:
    [https://github.com/jart/redisbayes] and [https://pypi.python.org/pypi/redisbayes]

    I've elected to write this to alleviate the network/time requirements when
    using the bayesian classifier to classify large sets of text, or when
    attempting to train with very large sets of sample data.


Installation
------------

PIP::

    sudo pip install simplebayes


GIT::

    sudo pip install git+git://github.com/hickeroar/simplebayes.git


Basic Usage
-----------

::

    import simplebayes
    bayes = simplebayes.SimpleBayes()

    bayes.train('good', 'sunshine drugs love sex lobster sloth')
    bayes.train('bad', 'fear death horror government zombie')

    assert bayes.classify('sloths are so cute i love them') == 'good'
    assert bayes.classify('i would fear a zombie and love the government') == 'bad'

    print bayes.score('i fear god and love the government')
