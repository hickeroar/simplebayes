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
