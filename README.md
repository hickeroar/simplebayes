simplebayes
===========
A memory-based, non-persistent naïve bayesian text classifier.
--------------------------------------------------------------
```
This work is heavily inspired by the python "redisbayes" module found here:
[https://github.com/jart/redisbayes] and [https://pypi.python.org/pypi/redisbayes]

I've elected to write this to alleviate the network/time requirements when
using the bayesian classifier to classify large sets of text, or when
attempting to train with very large sets of sample data.
```

Installation
------------
```
PIP:
sudo pip install simplebayes
```
```
GIT:
sudo pip install git+git://github.com/hickeroar/simplebayes.git
```

Basic Usage
-----------
```
import simplebayes
bayes = simplebayes.SimpleBayes()

bayes.train('good', 'sunshine drugs love sex lobster sloth')
bayes.train('bad', 'fear death horror government zombie')

assert bayes.classify('sloths are so cute i love them') == 'good'
assert bayes.classify('i would fear a zombie and love the government') == 'bad'

print bayes.score('i fear god and love the government')
```

License
-------
```
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
```

Python Package Requirements
---------------------------
1. xxhash
