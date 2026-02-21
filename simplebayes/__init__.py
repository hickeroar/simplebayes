# coding: utf-8
__version__ = '2.1.0'

import os
import pickle
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

from simplebayes.categories import BayesCategories
from simplebayes.errors import InvalidCategoryError
from simplebayes.models import CategorySummary, ClassificationResult
from simplebayes.tokenization import default_tokenize_text

__all__ = ['SimpleBayes']

CATEGORY_PATTERN = re.compile(r"^[-_A-Za-z0-9]{1,64}$")


class SimpleBayes:
    """A memory-based, optional-persistence naïve bayesian text classifier."""

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        cache_path: str = '/tmp/',
        cache_file: str = '_simplebayes.pickle',
    ) -> None:
        """
        :param tokenizer: A tokenizer override
        :type tokenizer: function (optional)
        :param cache_path: path to data storage
        :type cache_path: str
        :param cache_file: cache filename to persist/load
        :type cache_file: str
        """
        self.categories = BayesCategories()
        self.tokenizer = tokenizer or default_tokenize_text
        self.cache_path = cache_path
        self.cache_file = cache_file
        self.probabilities = {}
        self._lock = threading.RLock()

    @classmethod
    def tokenize_text(cls, text: str) -> List[str]:
        """
        Default tokenize method; can be overridden

        :param text: the text we want to tokenize
        :type text: str
        :return: list of tokenized text
        :rtype: list
        """
        return default_tokenize_text(text)

    @classmethod
    def count_token_occurrences(cls, words: List[str]) -> Dict[str, int]:
        """
        Creates a key/value set of word/count for a given sample of text

        :param words: full list of all tokens, non-unique
        :type words: list
        :return: key/value pairs of words and their counts in the list
        :rtype: dict
        """
        return dict(Counter(words))

    def flush(self) -> None:
        """
        Deletes all tokens & categories
        """
        with self._lock:
            self.categories = BayesCategories()
            self.probabilities = {}

    def calculate_category_probability(self) -> None:
        """
        Caches the individual probabilities for each category
        """
        with self._lock:
            total_tally = 0.0
            probs = {}
            for category, bayes_category in \
                    self.categories.get_categories().items():
                count = bayes_category.get_tally()
                total_tally += count
                probs[category] = count

            # Calculating the probability
            for category, count in probs.items():
                if total_tally > 0:
                    probs[category] = float(count)/float(total_tally)
                else:
                    probs[category] = 0.0

            for category, probability in probs.items():
                self.probabilities[category] = {
                    # Probability that any given token is of this category
                    'prc': probability,
                    # Probability that any given token is not of this category
                    'prnc': sum(probs.values()) - probability
                }

    def train(self, category: str, text: str) -> None:
        """
        Trains a category with a sample of text

        :param category: the name of the category we want to train
        :type category: str
        :param text: the text we want to train the category with
        :type text: str
        """
        category = self.normalize_category(category)
        with self._lock:
            try:
                bayes_category = self.categories.get_category(category)
            except KeyError:
                bayes_category = self.categories.add_category(category)

            tokens = self.tokenizer(str(text))
            occurrence_counts = self.count_token_occurrences(tokens)

            for word, count in occurrence_counts.items():
                bayes_category.train_token(word, count)

            # Updating our per-category overall probabilities
            self.calculate_category_probability()

    def untrain(self, category: str, text: str) -> None:
        """
        Untrains a category with a sample of text

        :param category: the name of the category we want to train
        :type category: str
        :param text: the text we want to untrain the category with
        :type text: str
        """
        category = self.normalize_category(category)
        with self._lock:
            try:
                bayes_category = self.categories.get_category(category)
            except KeyError:
                return

            tokens = self.tokenizer(str(text))
            occurrence_counts = self.count_token_occurrences(tokens)

            for word, count in occurrence_counts.items():
                bayes_category.untrain_token(word, count)

            if bayes_category.get_tally() == 0:
                self.categories.delete_category(category)

            # Updating our per-category overall probabilities
            self.calculate_category_probability()

    def classify(self, text: str) -> Optional[str]:
        """
        Chooses the highest scoring category for a sample of text

        :param text: sample text to classify
        :type text: str
        :return: the "winning" category
        :rtype: str
        """
        with self._lock:
            score = self.score(text)
            if not score:
                return None

            highest_category = ''
            highest_score = 0.0
            for category in sorted(score.keys()):
                category_score = float(score[category])
                if category_score > highest_score:
                    highest_score = category_score
                    highest_category = category

            return highest_category or None

    def classify_result(self, text: str) -> ClassificationResult:
        """
        Returns structured classification output including score.
        """
        with self._lock:
            scores = self.score(text)
            if not scores:
                return ClassificationResult(category=None, score=0.0)

            highest_category = ''
            highest_score = 0.0
            for category in sorted(scores.keys()):
                category_score = float(scores[category])
                if category_score > highest_score:
                    highest_score = category_score
                    highest_category = category

            return ClassificationResult(category=highest_category or None, score=highest_score)

    def score(self, text: str) -> Dict[str, float]:
        """
        Scores a sample of text

        :param text: sample text to score
        :type text: str
        :return: dict of scores per category
        :rtype: dict
        """
        with self._lock:
            occurs = self.count_token_occurrences(self.tokenizer(text))
            scores = {}
            for category in self.categories.get_categories().keys():
                scores[category] = 0

            categories = self.categories.get_categories().items()

            for word, count in occurs.items():
                token_scores = {}

                # Adding up individual token scores
                for category, bayes_category in categories:
                    token_scores[category] = \
                        float(bayes_category.get_token_count(word))

                # We use this to get token-in-category probabilities
                token_tally = sum(token_scores.values())

                # If this token isn't found anywhere its probability is 0
                if token_tally == 0.0:
                    continue

                # Calculating bayes probability for this token
                # http://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering
                for category, token_score in token_scores.items():
                    # Bayes probability * the number of occurrences of this token
                    scores[category] += count * \
                        self.calculate_bayesian_probability(
                            category,
                            token_score,
                            token_tally
                        )

            # Removing empty categories from the results
            final_scores = {}
            for category, score in scores.items():
                if score > 0:
                    final_scores[category] = score

            return final_scores

    def calculate_bayesian_probability(
        self, cat: str, token_score: float, token_tally: float
    ) -> float:
        """
        Calculates the bayesian probability for a given token/category

        :param cat: The category we're scoring for this token
        :type cat: str
        :param token_score: The tally of this token for this category
        :type token_score: float
        :param token_tally: The tally total for this token from all categories
        :type token_tally: float
        :return: bayesian probability
        :rtype: float
        """
        # P that any given token IS in this category
        prc = self.probabilities[cat]['prc']
        # P that any given token is NOT in this category
        prnc = self.probabilities[cat]['prnc']
        # P that this token is NOT of this category
        prtnc = (token_tally - token_score) / token_tally
        # P that this token IS of this category
        prtc = token_score / token_tally

        # Assembling the parts of the bayes equation
        numerator = (prtc * prc)
        denominator = (numerator + (prtnc * prnc))

        # Returning the calculated bayes probability unless the denom. is 0
        return numerator / denominator if denominator != 0.0 else 0.0

    def tally(self, category: str) -> int:
        """
        Gets the tally for a requested category

        :param category: The category we want a tally for
        :type category: str
        :return: tally for a given category
        :rtype: int
        """
        with self._lock:
            try:
                bayes_category = self.categories.get_category(category)
            except KeyError:
                return 0

            return bayes_category.get_tally()

    def get_summaries(self) -> Dict[str, CategorySummary]:
        """
        Returns per-category summary details.
        """
        with self._lock:
            summaries: Dict[str, CategorySummary] = {}
            categories = self.categories.get_categories()

            for category_name, category in categories.items():
                category_probability = self.probabilities.get(
                    category_name,
                    {'prc': 0.0, 'prnc': 0.0},
                )
                summaries[category_name] = CategorySummary(
                    token_tally=category.get_tally(),
                    prob_in_cat=float(category_probability['prc']),
                    prob_not_in_cat=float(category_probability['prnc']),
                )

            return summaries

    def get_cache_location(self) -> str:
        """
        Gets the location of the cache file

        :return: the location of the cache file
        :rtype: string
        """
        return str(Path(self.cache_path) / self.cache_file)

    def cache_persist(self) -> None:
        """
        Saves the current trained data to the cache.
        This is initiated by the program using this module
        """
        with self._lock:
            filename = self.get_cache_location()
            with open(filename, 'wb') as cache_file:
                pickle.dump(self.categories, cache_file)

    def cache_train(self) -> bool:
        """
        Loads the data for this classifier from a cache file

        :return: whether or not we were successful
        :rtype: bool
        """
        with self._lock:
            filename = self.get_cache_location()

            if not os.path.exists(filename):
                return False

            with open(filename, 'rb') as cache_file:
                categories = pickle.load(cache_file)

            assert isinstance(categories, BayesCategories), \
                "Cache data is either corrupt or invalid"

            self.categories = categories

            # Updating our per-category overall probabilities
            self.calculate_category_probability()

            return True

    @classmethod
    def normalize_category(cls, category: str) -> str:
        """
        Validates and normalizes category input.
        """
        if category is None:
            raise InvalidCategoryError("category is required")

        normalized = str(category).strip()
        if not CATEGORY_PATTERN.match(normalized):
            raise InvalidCategoryError(
                "category must be 1-64 chars and only include letters, numbers, underscore, or hyphen",
            )

        return normalized
