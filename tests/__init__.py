# pylint: disable=invalid-name,missing-docstring
import unittest
from unittest.mock import patch, MagicMock

from simplebayes import SimpleBayes
from simplebayes.categories import BayesCategories

from simplebayes.errors import InvalidCategoryError


class SimpleBayesTests(unittest.TestCase):

    def test_tokenizer(self):
        sb = SimpleBayes()
        result = sb.tokenizer('hello world')
        self.assertEqual(result, ['hello', 'world'])
        self.assertEqual(SimpleBayes.tokenize_text('hello world'), ['hello', 'world'])

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

    @patch.object(BayesCategories, 'get_category')
    def test_train_with_existing_category(self, get_category_mock):
        cat_mock = MagicMock()
        cat_mock.train_token.return_value = None
        get_category_mock.return_value = cat_mock

        sb = SimpleBayes()
        sb.train('foo', 'hello world hello')

        get_category_mock.assert_called_once_with('foo')
        cat_mock.train_token.assert_any_call('hello', 2)
        cat_mock.train_token.assert_any_call('world', 1)

    @patch.object(BayesCategories, 'get_category')
    @patch.object(BayesCategories, 'add_category')
    def test_train_with_new_category(
            self,
            add_category_mock,
            get_category_mock
    ):
        cat_mock = MagicMock()
        cat_mock.train_token.return_value = None
        get_category_mock.side_effect = KeyError()
        add_category_mock.return_value = cat_mock

        sb = SimpleBayes()
        sb.train('foo', 'hello world hello')

        add_category_mock.assert_called_with('foo')
        cat_mock.train_token.assert_any_call('hello', 2)
        cat_mock.train_token.assert_any_call('world', 1)

    @patch.object(BayesCategories, 'get_categories')
    def test_classify(self, get_categories_mock):
        cat1_mock = MagicMock()
        cat1_mock.get_token_count.return_value = 2
        cat1_mock.get_tally.return_value = 8
        cat2_mock = MagicMock()
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

    @patch.object(BayesCategories, 'get_categories')
    def test_classify_without_categories(self, get_categories_mock):
        get_categories_mock.return_value = {}

        sb = SimpleBayes()
        result = sb.classify('hello world')

        self.assertIsNone(result)
        assert 2 == get_categories_mock.call_count, \
            get_categories_mock.call_count

    @patch.object(BayesCategories, 'get_categories')
    def test_classify_with_empty_category(self, get_categories_mock):
        cat_mock = MagicMock()
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

    def test_score_without_categories(self):
        sb = SimpleBayes()
        self.assertEqual(sb.score('hello world'), {})

    def test_score_with_no_matching_tokens(self):
        sb = SimpleBayes()
        sb.train('alpha', 'one two three')
        self.assertEqual(sb.score('unknown tokens here'), {})

    @patch.object(BayesCategories, 'get_categories')
    def test_score(self, get_categories_mock):
        cat1_mock = MagicMock()
        cat1_mock.get_token_count.return_value = 2
        cat1_mock.get_tally.return_value = 8
        cat2_mock = MagicMock()
        cat2_mock.get_token_count.return_value = 4
        cat2_mock.get_tally.return_value = 32

        get_categories_mock.return_value = {
            'foo': cat1_mock,
            'bar': cat2_mock
        }

        sb = SimpleBayes()
        sb.calculate_category_probability()
        result = sb.score('hello world')

        self.assertIn('foo', result)
        self.assertIn('bar', result)
        self.assertAlmostEqual(result['foo'], 0.22222222222222224)
        self.assertAlmostEqual(result['bar'], 1.777777777777778)

        assert 3 == get_categories_mock.call_count, \
            get_categories_mock.call_count
        cat1_mock.get_token_count.assert_any_call('hello')
        cat1_mock.get_token_count.assert_any_call('world')
        cat1_mock.get_tally.assert_called_once_with()
        cat2_mock.get_token_count.assert_any_call('hello')
        cat2_mock.get_token_count.assert_any_call('world')
        cat2_mock.get_tally.assert_called_once_with()

    @patch.object(BayesCategories, 'get_categories')
    def test_score_with_zero_bayes_denon(self, get_categories_mock):
        cat1_mock = MagicMock()
        cat1_mock.get_token_count.return_value = 2
        cat1_mock.get_tally.return_value = 8
        cat2_mock = MagicMock()
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

    def test_classify_result(self):
        sb = SimpleBayes()
        sb.train('good', 'bright happy joy')
        sb.train('bad', 'sad dark doom')

        result = sb.classify_result('bright joy')

        self.assertEqual(result.category, 'good')
        self.assertGreater(result.score, 0)

    def test_classify_result_empty(self):
        sb = SimpleBayes()

        result = sb.classify_result('anything')

        self.assertIsNone(result.category)
        self.assertEqual(result.score, 0.0)

    def test_get_summaries(self):
        sb = SimpleBayes()
        sb.train('alpha', 'one two three')

        summaries = sb.get_summaries()

        self.assertIn('alpha', summaries)
        self.assertEqual(summaries['alpha'].token_tally, 3)
        self.assertGreaterEqual(summaries['alpha'].prob_in_cat, 0.0)
        self.assertGreaterEqual(summaries['alpha'].prob_not_in_cat, 0.0)

    def test_train_invalid_category_raises(self):
        sb = SimpleBayes()
        with self.assertRaises(InvalidCategoryError):
            sb.train('bad category', 'text')
        with self.assertRaises(InvalidCategoryError):
            sb.train(None, 'text')  # type: ignore[arg-type]

    def test_untrain_removes_empty_category(self):
        sb = SimpleBayes()
        sb.train('alpha', 'one two three')
        sb.untrain('alpha', 'one two three')
        self.assertNotIn('alpha', sb.categories.get_categories())
        self.assertNotIn('alpha', sb.probabilities)
        self.assertNotIn('alpha', sb.get_summaries())

    def test_classify_tie_breaks_lexically(self):
        sb = SimpleBayes()
        sb.train('zeta', 'match token')
        sb.train('alpha', 'match token')

        result = sb.classify('match token')

        self.assertEqual(result, 'alpha')
