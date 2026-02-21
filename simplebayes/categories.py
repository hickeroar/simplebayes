from typing import Dict

from simplebayes.category import BayesCategory


class BayesCategories:
    """Acts as a container for various bayes trained categories of content"""

    def __init__(self):
        self.categories: Dict[str, BayesCategory] = {}

    def add_category(self, name: str) -> BayesCategory:
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

    def get_category(self, name: str) -> BayesCategory:
        """
        Returns the expected category. Will KeyError if non existent

        :param name: name of the category
        :type name: str
        :return: the requested category
        :rtype: BayesCategory
        """
        return self.categories[name]

    def get_categories(self) -> Dict[str, BayesCategory]:
        """
        :return: dict of all categories
        :rtype: dict
        """
        return self.categories

    def delete_category(self, name: str) -> None:
        """
        Deletes an existing category when present.

        :param name: name of the category
        :type name: str
        """
        self.categories.pop(name, None)
