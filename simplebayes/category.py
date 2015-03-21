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
import xxhash


class BayesCategory(object):

    def __init__(self, name):
        self.name = name
        self.tokens = {}
        self.tally = 0

    @classmethod
    def get_word_name(cls, word):
        return xxhash.xxh32(word).hexdigest()

    def train_token(self, word, count):
        word = self.get_word_name(word)
        if word not in self.tokens:
            self.tokens[word] = 0
        self.tokens[word] += count
        self.tally += count

    def get_token_count(self, word):
        word = self.get_word_name(word)
        try:
            return self.tokens[word]
        except KeyError:
            return 0

    def get_tally(self):
        """Gets the tally of all types"""
        return self.tally
