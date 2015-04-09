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


class BayesCategory(object):
    """
    Represents a trainable category of content for bayesian classification
    """

    def __init__(self, name):
        """
        :param name: The name of the category we're creating
        :type name: str
        """
        self.name = name
        self.tokens = {}
        self.tally = 0

    def train_token(self, word, count):
        """
        Trains a particular token (increases the weight/count of it)

        :param word: the token we're going to train
        :type word: str
        :param count: the number of occurances in the sample
        :type count: int
        """
        if word not in self.tokens:
            self.tokens[word] = 0

        self.tokens[word] += count
        self.tally += count

    def untrain_token(self, word, count):
        """
        Untrains a particular token (decreases the weight/count of it)

        :param word: the token we're going to train
        :type word: str
        :param count: the number of occurances in the sample
        :type count: int
        """
        if word not in self.tokens:
            return

        # If we're trying to untrain more tokens than we have, we end at 0
        count = min(count, self.tokens[word])

        self.tokens[word] -= count
        self.tally -= count

    def get_token_count(self, word):
        """
        Gets the count assosicated with a provided token/word

        :param word: the token we're getting the weight of
        :type word: str
        :return: the weight/count of the token
        :rtype: int
        """
        return self.tokens.get(word, 0)

    def get_tally(self):
        """
        Gets the tally of all types

        :return: The total number of tokens
        :rtype: int
        """
        return self.tally
