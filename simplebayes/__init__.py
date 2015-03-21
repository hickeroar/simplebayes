"""
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

------------------------------------------------------------------------------

This work is heavily inspired by the python "redisbayes" module found here:
https://github.com/jart/redisbayes
 and
https://pypi.python.org/pypi/redisbayes/0.1.3

I've elected to rewrite this to alleviate the network/time requirements when
using the bayesian classifier to classify large sets of text, or when
attempting to train with very large sets of sample data.
"""
import math
from simplebayes.categories import BayesCategories


class SimpleBayes(object):
    """A naïve bayesian text classifier in memory."""

    def __init__(self, tokenizer=None):
        self.categories = BayesCategories()
        self.tokenizer = tokenizer or SimpleBayes.tokenize_text

    @classmethod
    def tokenize_text(cls, text):
        """Default tokenize method; can be overridden"""
        return [w for w in text.split() if len(w) > 2]

    @classmethod
    def count_token_occurrences(cls, words):
        """creates a key/value set of word/count for a given sample of text"""
        counts = {}
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        return counts

    def train(self, category, text):
        """Trains a category with a sample of text"""
        try:
            bayes_category = self.categories.get_category(category)
        except KeyError:
            bayes_category = self.categories.add_category(category)

        tokens = self.tokenizer(text)
        occurrence_counts = self.count_token_occurrences(tokens)

        for word, count in occurrence_counts.iteritems():
            bayes_category.train_token(word, count)

    def classify(self, text):
        """Chooses the highest scoring category for a sample of text"""
        score = self.score(text)
        if not score:
            return None
        return sorted(score.iteritems(), key=lambda v: v[1])[-1][0]

    def score(self, text):
        """Scores a sample of text"""
        occurs = self.count_token_occurrences(self.tokenizer(text))
        scores = {}
        for category, bayes_category in self.categories.get_categories().iteritems():
            if bayes_category.get_tally() == 0:
                continue
            scores[category] = 0.0
            for word, count in occurs.iteritems():
                score = bayes_category.get_token_count(word) or 0.1
                scores[category] += math.log(float(score) / bayes_category.get_tally())
        return scores
