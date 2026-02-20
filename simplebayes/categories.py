from simplebayes.category import BayesCategory


class BayesCategories:
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
        Returns the expected category. Will KeyError if non existent

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
