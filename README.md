# simplebayes

A memory-based, optional-persistence naïve bayesian text classifier.

This work is heavily inspired by the python "redisbayes" module found here:
[https://github.com/jart/redisbayes](https://github.com/jart/redisbayes) and [https://pypi.python.org/pypi/redisbayes](https://pypi.python.org/pypi/redisbayes)

I've elected to write this to alleviate the network/time requirements when using the bayesian classifier to classify large sets of text, or when attempting to train with very large sets of sample data.

## Build Status

![Build](https://github.com/hickeroar/simplebayes/actions/workflows/test.yml/badge.svg?branch=master)
![Coverage](https://img.shields.io/badge/coverage-100%-brightgreen.svg?style=flat)
![Pylint](https://img.shields.io/badge/pylint-10.00/10-brightgreen.svg?style=flat)
![Flake8](https://img.shields.io/badge/flake8-passing-brightgreen.svg?style=flat)

## Installation

```bash
pip install simplebayes
```

## Basic Usage

```python
import simplebayes
bayes = simplebayes.SimpleBayes()

bayes.train('good', 'sunshine drugs love sex lobster sloth')
bayes.train('bad', 'fear death horror government zombie')

assert bayes.classify('sloths are so cute i love them') == 'good'
assert bayes.classify('i would fear a zombie and love the government') == 'bad'

print(bayes.score('i fear zombies and love the government'))

bayes.untrain('bad', 'fear death')

assert bayes.tally('bad') == 3
```

## Cache Usage

```python
import simplebayes
bayes = simplebayes.SimpleBayes(
    cache_path='/my/cache/',
    cache_file='project-a.pickle',
)
# Cache file is '/my/cache/project-a.pickle'
# Default cache_path is '/tmp/'
# Default cache_file is '_simplebayes.pickle'

if not bayes.cache_train():
    # Unable to load cache data, so we're training it
    bayes.train('good', 'sunshine drugs love sex lobster sloth')
    bayes.train('bad', 'fear death horror government zombie')

    # Saving the cache so next time the training won't be needed
    bayes.cache_persist()
```

Use different `cache_file` values when running multiple `SimpleBayes` objects
to avoid cache collisions.

## Tokenizer Override

```python
import simplebayes

def my_tokenizer(sample):
    return sample.split()

bayes = simplebayes.SimpleBayes(tokenizer=my_tokenizer)
```

## License

MIT, see `LICENSE`.

## API Documentation

[http://hickeroar.github.io/simplebayes/simplebayes.html](http://hickeroar.github.io/simplebayes/simplebayes.html)
