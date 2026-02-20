#!/usr/bin/python
# pylint: disable=unused-wildcard-import,wildcard-import,unused-import
# Test aggregator: star imports register test classes (F401/F403 in .flake8)

import simplebayes
import simplebayes.categories
import simplebayes.category

from tests import *
from tests.categories import *
from tests.category import *
