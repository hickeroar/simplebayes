#!/bin/bash
echo
echo
echo " [simplebayes] Step 1: Executing Unit Tests (Python 2)"
echo
nosetests tests/test.py --with-coverage --cover-package=simplebayes --cover-min-percentage 100
rm -f .coverage*
echo -e "\nExit Code:" $?
echo
echo " [simplebayes] Step 1: Executing Unit Tests (Python 3)"
echo
nosetests3 tests/test.py --with-coverage --cover-package=simplebayes --cover-min-percentage 100
rm -f .coverage*
echo -e "\nExit Code:" $?

echo
echo " [simplebayes] Step 2: Executing pep8 and pyflakes Tests (flake8). (Python 2)"
echo
flake8 simplebayes tests
echo "Exit Code:" $?
echo
echo " [simplebayes] Step 2: Executing pep8 and pyflakes Tests (flake8). (Python 3)"
echo
flake83 simplebayes tests
echo "Exit Code:" $?

echo
echo " [simplebayes] Step 3: Executing pylint Tests (Python 2)"
echo
pylint simplebayes tests --reports=no
echo "Exit Code:" $?
echo
echo " [simplebayes] Step 3: Executing pylint Tests (Python 3)"
echo
pylint3 simplebayes tests --reports=no
echo "Exit Code:" $?
echo
