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
import os


class SimpleBayes(object):
    """A memory-based, optional-persistence naïve bayesian text classifier."""

    cache_file = '_simplebayes.pickle'

    def __init__(self, tokenizer=None, cache_path='/tmp/'):
        self.categories = BayesCategories()
        self.tokenizer = tokenizer or SimpleBayes.tokenize_text
        self.cache_path = cache_path
        self.probabilities = {}

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

    def flush(self):
        """Deletes all tokens & categories"""
        self.categories = BayesCategories()

    def calculate_category_prob(self):
        """Caches the individual probabilities for each category"""
        total_tally = 0.0
        probs = {}
        for category, bayes_category in \
                self.categories.get_categories().items():
            count = bayes_category.get_tally()
            total_tally += count
            probs[category] = count

        for category, count in probs.items():
            if total_tally > 0:
                probs[category] = count/total_tally
            else:
                probs[category] = 0

        for category, _ in \
                self.categories.get_categories().items():
            self.probabilities[category] = {
                # Probability that any given token is of this category
                'prc': probs[category],
                # Probability that any given token is not of this category
                'prnc': sum(probs.values()) - probs[category]
            }

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

        tokens = self.tokenizer(str(text))
        occurrence_counts = self.count_token_occurrences(tokens)

        for word, count in occurrence_counts.items():
            bayes_category.train_token(word, count)

        # Updating our per-category overall probabilities
        self.calculate_category_prob()

    def untrain(self, category, text):
        """
        Untrains a category with a sample of text
        :param category: the name of the category we want to train
        :type category: str
        :param text: the text we want to untrain the category with
        :type text: str
        """
        try:
            bayes_category = self.categories.get_category(category)
        except KeyError:
            return

        tokens = self.tokenizer(str(text))
        occurance_counts = self.count_token_occurrences(tokens)

        for word, count in occurance_counts.items():
            bayes_category.untrain_token(word, count)

        # Updating our per-category overall probabilities
        self.calculate_category_prob()

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
        for category in self.categories.get_categories().keys():
            scores[category] = 0

        for word, count in occurs.items():
            token_scores = {}

            # Calculating individual token probabilities
            for category, bayes_category in \
                    self.categories.get_categories().items():
                token_scores[category] = bayes_category.get_token_count(word)

            # Calculating bayes probabiltity for this token
            # http://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering
            for category, token_score in token_scores.items():
                # P that this token is NOT of this category
                prtnc = sum(token_scores.values()) - token_score

                # Assembling the parts of the bayes equation
                numerator = (token_score * self.probabilities[category]['prc'])
                denominator = (
                    (token_score * self.probabilities[category]['prc']) +
                    (prtnc * self.probabilities[category]['prnc'])
                )

                if denominator == 0.0:
                    continue

                # Bayes probability calculation
                scores[category] += count * (numerator / denominator)

        # Removing empty categories from the results
        final_scores = {}
        for category, score in scores.items():
            if score > 0:
                final_scores[category] = score

        return final_scores

    def tally(self, category):
        """Gets the tally for a requested category"""
        try:
            bayes_category = self.categories.get_category(category)
        except KeyError:
            return 0

        return bayes_category.get_tally()

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

        if not os.path.exists(filename):
            return False

        categories = pickle.load(open(filename, 'rb'))

        assert isinstance(categories, BayesCategories), \
            "Cache data is either corrupt or invalid"

        self.categories = categories

        return True
