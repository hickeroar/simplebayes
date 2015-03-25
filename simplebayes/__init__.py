# coding: utf-8
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
"""
from simplebayes.categories import BayesCategories
import pickle
import math


class SimpleBayes(object):
    """A memory-based, optional-persistence naïve bayesian text classifier."""

    cache_file = '_simplebayes.pickle'

    def __init__(self, tokenizer=None, cache_path=None):
        self.categories = BayesCategories()
        self.tokenizer = tokenizer or SimpleBayes.tokenize_text

        if cache_path is not None:
            self.cache_path = cache_path
            self.cache_train()
        else:
            self.cache_path = '/tmp/'

    @classmethod
    def tokenize_text(cls, text):
        """
        Default tokenize method; can be overridden
        :param text: the text we want to tokenize
        :type text: str
        :return: list of tokenized text
        :rtype: list
        """
        return [w for w in text.split() if len(w) > 2]

    @classmethod
    def count_token_occurrences(cls, words):
        """
        Creates a key/value set of word/count for a given sample of text
        :param words: full list of all tokens, non-unique
        :type words: list
        :return: key/value pairs of words and their counts in the list
        :rtype: dict
        """
        counts = {}
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        return counts

    def train(self, category, text):
        """
        Trains a category with a sample of text
        :param category: the name of the category we want to train
        :type category: str
        :param text: the text we want to train the category with
        :type text: str
        """
        try:
            bayes_category = self.categories.get_category(category)
        except KeyError:
            bayes_category = self.categories.add_category(category)

        tokens = self.tokenizer(text)
        occurrence_counts = self.count_token_occurrences(tokens)

        for word, count in occurrence_counts.items():
            bayes_category.train_token(word, count)

    def classify(self, text):
        """
        Chooses the highest scoring category for a sample of text
        :param text: sample text to classify
        :type text: str
        :return: the "winning" category
        :rtype: str
        """
        score = self.score(text)
        if not score:
            return None
        return sorted(score.items(), key=lambda v: v[1])[-1][0]

    def score(self, text):
        """
        Scores a sample of text
        :param text: sample text to score
        :type text: str
        :return: dict of scores per category
        :rtype: dict
        """
        occurs = self.count_token_occurrences(self.tokenizer(text))
        scores = {}
        for category, bayes_category in \
                self.categories.get_categories().items():
            category_tally = bayes_category.get_tally()
            if category_tally == 0:
                continue
            scores[category] = 0.0
            for word, _ in occurs.items():
                score = bayes_category.get_token_count(word) or 0.1
                scores[category] += \
                    math.log(float(score) / category_tally)
        return scores

    def get_cache_location(self):
        """
        Gets the location of the cache file
        """
        filename = self.cache_path if \
            self.cache_path[-1:] == '/' else \
            self.cache_path + '/'
        filename += self.cache_file
        return filename

    def cache_persist(self):
        """
        Saves the current trained data to the cache.
        This is initiated by the program using this module
        """
        filename = self.get_cache_location()
        pickle.dump(self.categories, open(filename, 'wb'))

    def cache_train(self):
        """
        Loads the data for this classifier from a cache file
        """
        filename = self.get_cache_location()
        categories = pickle.load(open(filename, 'rb'))

        assert isinstance(categories, BayesCategories), \
            "Cache data is either corrupt or invalid"

        self.categories = categories
