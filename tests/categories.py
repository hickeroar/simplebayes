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
# pylint: disable=invalid-name,missing-docstring,no-self-use
from simplebayes.categories import BayesCategories
from simplebayes.category import BayesCategory
import unittest


class BayesCategoriesTests(unittest.TestCase):

    def test_add_category(self):
        bc = BayesCategories()
        bc.add_category('foo')
        self.assertIn('foo', bc.categories)
        self.assertIsInstance(bc.categories['foo'], BayesCategory)

    def test_get_category(self):
        bc = BayesCategories()
        bc.add_category('foo')
        self.assertIsInstance(bc.get_category('foo'), BayesCategory)

    def test_get_categories(self):
        bc = BayesCategories()
        bc.add_category('foo')
        self.assertEqual(bc.get_categories(), bc.categories)
