# pylint: disable=invalid-name,missing-docstring
from simplebayes.category import BayesCategory
import unittest


class BayesCategoryTests(unittest.TestCase):

    def test_train_token(self):
        bc = BayesCategory('foo')
        bc.train_token('foo', 5)
        bc.train_token('bar', 7)
        self.assertEqual(12, bc.tally)
        self.assertIn('foo', bc.tokens)
        self.assertEqual(bc.tokens['foo'], 5)

    def test_untrain_token(self):
        bc = BayesCategory('foo')
        bc.train_token('foo', 5)
        bc.train_token('bar', 7)
        self.assertEqual(12, bc.tally)
        self.assertIn('foo', bc.tokens)
        self.assertIn('bar', bc.tokens)
        self.assertEqual(bc.tokens['foo'], 5)
        self.assertEqual(bc.tokens['bar'], 7)
        bc.untrain_token('foo', 3)
        bc.untrain_token('bar', 20)
        bc.untrain_token('baz', 5)
        self.assertEqual(2, bc.tally)
        self.assertEqual(bc.tokens['foo'], 2)
        self.assertNotIn('bar', bc.tokens)

    def test_get_token_count(self):
        bc = BayesCategory('foo')
        bc.train_token('foo', 5)
        self.assertEqual(bc.get_token_count('foo'), 5)
        self.assertEqual(bc.get_token_count('bar'), 0)

    def test_get_tally(self):
        bc = BayesCategory('foo')
        bc.train_token('foo', 5)
        self.assertEqual(5, bc.get_tally())
