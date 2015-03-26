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

Installation
------------
::

    sudo pip install simplebayes

Basic Usage
-----------
::

    import simplebayes
    bayes = simplebayes.SimpleBayes()

    bayes.train('good', 'sunshine drugs love sex lobster sloth')
    bayes.train('bad', 'fear death horror government zombie')

    assert bayes.classify('sloths are so cute i love them') == 'good'
    assert bayes.classify('i would fear a zombie and love the government') == 'bad'

    print bayes.score('i fear zombies and love the government')

Cache Usage
-----------
::

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
::

    import simplebayes

    def my_tokenizer(sample):
        return sample.split()

    bayes = simplebayes.SimpleBayes(tokenizer=my_tokenizer)
