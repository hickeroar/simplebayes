#!/bin/bash
echo
echo
echo " [simplebayes] Step 1: Executing Unit Tests"
echo
nosetests3 tests/test.py --with-coverage --cover-package=simplebayes --cover-min-percentage 100 -s
rm -f .coverage*
echo -e "\nExit Code:" $?

echo
echo " [simplebayes] Step 2: Executing pep8 and pyflakes Tests (flake8)."
echo
flake8 simplebayes tests
echo "Exit Code:" $?

echo
echo " [simplebayes] Step 3: Executing pylint Tests"
echo
pylint simplebayes tests --reports=no
echo "Exit Code:" $?
echo
