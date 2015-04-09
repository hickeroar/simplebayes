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
from simplebayes.category import BayesCategory


class BayesCategories(object):
    """Acts as a container for various bayes trained categories of content"""

    def __init__(self):
        self.categories = {}

    def add_category(self, name):
        """
        Adds a bayes category that we can later train

        :param name: name of the category
        :type name: str
        :return: the requested category
        :rtype: BayesCategory
        """
        category = BayesCategory(name)
        self.categories[name] = category
        return category

    def get_category(self, name):
        """
        Returns the expected category. Will KeyError if non existant

        :param name: name of the category
        :type name: str
        :return: the requested category
        :rtype: BayesCategory
        """
        return self.categories[name]

    def get_categories(self):
        """
        :return: dict of all categories
        :rtype: dict
        """
        return self.categories
