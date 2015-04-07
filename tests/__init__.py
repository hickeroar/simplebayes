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
from simplebayes import SimpleBayes
from simplebayes.categories import BayesCategories
import unittest
import mock
import pickle
import os
try:
    import __builtin__ as builtins
except ImportError:
    # pylint: disable=import-error
    import builtins


class SimpleBayesTests(unittest.TestCase):

    def test_tokenizer(self):
        sb = SimpleBayes()
        result = sb.tokenizer('hello world')
        self.assertEqual(result, ['hello', 'world'])

    def test_count_token_occurrences(self):
        sb = SimpleBayes()
        result = sb.count_token_occurrences(['hello', 'world', 'hello'])
        self.assertEqual(
            result,
            {
                'hello': 2,
                'world': 1
            }
        )

    def test_flush_and_tally(self):
        sb = SimpleBayes()
        sb.train('foo', 'hello world hello')
        self.assertEqual(sb.tally('foo'), 3)
        sb.flush()
        self.assertEqual(sb.tally('foo'), 0)

    def test_untrain(self):
        sb = SimpleBayes()
        sb.train('foo', 'hello world hello')
        self.assertEqual(sb.tally('foo'), 3)
        self.assertEqual(sb.tally('bar'), 0)
        sb.untrain('bar', 'for bar baz')
        self.assertEqual(sb.tally('foo'), 3)
        self.assertEqual(sb.tally('bar'), 0)
        sb.untrain('foo', 'hello world')
        self.assertEqual(sb.tally('foo'), 1)

    @mock.patch.object(BayesCategories, 'get_category')
    # pylint: disable=no-self-use
    def test_train_with_existing_category(self, get_category_mock):
        cat_mock = mock.MagicMock()
        cat_mock.train_token.return_value = None
        get_category_mock.return_value = cat_mock

        sb = SimpleBayes()
        sb.train('foo', 'hello world hello')

        get_category_mock.assert_called_once_with('foo')
        cat_mock.train_token.assert_any_call('hello', 2)
        cat_mock.train_token.assert_any_call('world', 1)

    @mock.patch.object(BayesCategories, 'get_category')
    @mock.patch.object(BayesCategories, 'add_category')
    # pylint: disable=no-self-use
    def test_train_with_new_category(
            self,
            add_category_mock,
            get_category_mock
    ):
        cat_mock = mock.MagicMock()
        cat_mock.train_token.return_value = None
        get_category_mock.side_effect = KeyError()
        add_category_mock.return_value = cat_mock

        sb = SimpleBayes()
        sb.train('foo', 'hello world hello')

        add_category_mock.assert_called_with('foo')
        cat_mock.train_token.assert_any_call('hello', 2)
        cat_mock.train_token.assert_any_call('world', 1)

    @mock.patch.object(BayesCategories, 'get_categories')
    def test_classify(self, get_categories_mock):
        cat1_mock = mock.MagicMock()
        cat1_mock.get_token_count.return_value = 2
        cat1_mock.get_tally.return_value = 8
        cat2_mock = mock.MagicMock()
        cat2_mock.get_token_count.return_value = 4
        cat2_mock.get_tally.return_value = 32

        get_categories_mock.return_value = {
            'foo': cat1_mock,
            'bar': cat2_mock
        }

        sb = SimpleBayes()
        sb.calculate_category_probability()
        result = sb.classify('hello world')

        self.assertEqual('bar', result)
        assert 3 == get_categories_mock.call_count, \
            get_categories_mock.call_count
        cat1_mock.get_token_count.assert_any_call('hello')
        cat1_mock.get_token_count.assert_any_call('world')
        cat1_mock.get_tally.assert_called_once_with()
        cat2_mock.get_token_count.assert_any_call('hello')
        cat2_mock.get_token_count.assert_any_call('world')
        cat2_mock.get_tally.assert_called_once_with()

    @mock.patch.object(BayesCategories, 'get_categories')
    def test_classify_without_categories(self, get_categories_mock):
        get_categories_mock.return_value = {}

        sb = SimpleBayes()
        result = sb.classify('hello world')

        self.assertIsNone(result)
        assert 2 == get_categories_mock.call_count, \
            get_categories_mock.call_count

    @mock.patch.object(BayesCategories, 'get_categories')
    def test_classify_with_empty_category(self, get_categories_mock):
        cat_mock = mock.MagicMock()
        cat_mock.get_tally.return_value = 0
        cat_mock.get_token_count.return_value = 0

        get_categories_mock.return_value = {
            'foo': cat_mock
        }

        sb = SimpleBayes()
        sb.calculate_category_probability()
        result = sb.classify('hello world')

        self.assertIsNone(result)
        assert 3 == get_categories_mock.call_count, \
            get_categories_mock.call_count
        cat_mock.get_tally.assert_called_once_with()

    @mock.patch.object(BayesCategories, 'get_categories')
    def test_score(self, get_categories_mock):
        cat1_mock = mock.MagicMock()
        cat1_mock.get_token_count.return_value = 2
        cat1_mock.get_tally.return_value = 8
        cat2_mock = mock.MagicMock()
        cat2_mock.get_token_count.return_value = 4
        cat2_mock.get_tally.return_value = 32

        get_categories_mock.return_value = {
            'foo': cat1_mock,
            'bar': cat2_mock
        }

        sb = SimpleBayes()
        sb.calculate_category_probability()
        result = sb.score('hello world')

        self.assertEqual(
            {
                'foo': 0.22222222222222224,
                'bar': 1.777777777777778
            },
            result
        )

        assert 3 == get_categories_mock.call_count, \
            get_categories_mock.call_count
        cat1_mock.get_token_count.assert_any_call('hello')
        cat1_mock.get_token_count.assert_any_call('world')
        cat1_mock.get_tally.assert_called_once_with()
        cat2_mock.get_token_count.assert_any_call('hello')
        cat2_mock.get_token_count.assert_any_call('world')
        cat2_mock.get_tally.assert_called_once_with()

    @mock.patch.object(BayesCategories, 'get_categories')
    def test_score_with_zero_bayes_denon(self, get_categories_mock):
        cat1_mock = mock.MagicMock()
        cat1_mock.get_token_count.return_value = 2
        cat1_mock.get_tally.return_value = 8
        cat2_mock = mock.MagicMock()
        cat2_mock.get_token_count.return_value = 4
        cat2_mock.get_tally.return_value = 32

        get_categories_mock.return_value = {
            'foo': cat1_mock,
            'bar': cat2_mock
        }

        sb = SimpleBayes()
        sb.calculate_category_probability()
        sb.probabilities['foo']['prc'] = 0
        sb.probabilities['foo']['prnc'] = 0
        result = sb.score('hello world')

        self.assertEqual(
            {
                'bar': 1.777777777777778
            },
            result
        )

        assert 3 == get_categories_mock.call_count, \
            get_categories_mock.call_count
        cat1_mock.get_token_count.assert_any_call('hello')
        cat1_mock.get_token_count.assert_any_call('world')
        cat1_mock.get_tally.assert_called_once_with()
        cat2_mock.get_token_count.assert_any_call('hello')
        cat2_mock.get_token_count.assert_any_call('world')
        cat2_mock.get_tally.assert_called_once_with()

    @mock.patch.object(builtins, 'open')
    @mock.patch.object(pickle, 'load')
    @mock.patch.object(os.path, 'exists')
    def test_cache_train(self, exists_mock, load_mock, open_mock):
        categories = BayesCategories()
        categories.categories = {'foo': 'bar'}

        load_mock.return_value = categories
        open_mock.return_value = 'opened'
        exists_mock.return_value = True

        sb = SimpleBayes(cache_path='foo')
        sb.cache_train()

        exists_mock.assert_called_once_with('foo/_simplebayes.pickle')
        open_mock.assert_called_once_with('foo/_simplebayes.pickle', 'rb')
        load_mock.assert_called_once_with('opened')

        self.assertEqual(sb.categories, categories)

    @mock.patch.object(os.path, 'exists')
    def test_cache_train_with_no_file(self, exists_mock):
        exists_mock.return_value = False

        sb = SimpleBayes()
        result = sb.cache_train()

        exists_mock.assert_called_once_with('/tmp/_simplebayes.pickle')
        self.assertFalse(result)

    @mock.patch.object(builtins, 'open')
    @mock.patch.object(pickle, 'dump')
    def test_persist_cache(self, dump_mock, open_mock):
        open_mock.return_value = 'opened'

        categories = BayesCategories()
        categories.categories = {'foo': 'bar'}

        sb = SimpleBayes()
        sb.cache_path = '/tmp/'
        sb.categories = categories
        sb.cache_persist()

        open_mock.assert_called_once_with('/tmp/_simplebayes.pickle', 'wb')
        dump_mock.assert_called_once_with(categories, 'opened')
