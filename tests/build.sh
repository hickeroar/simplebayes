#!/bin/bash
echo
echo
echo " [Cahoots] Step 1: Executing Unit Tests"
echo
nosetests tests/test.py --with-coverage --cover-package=simplebayes --cover-min-percentage 100 -s
rm -f .coverage*
echo -e "\nExit Code:" $?

echo
echo " [Cahoots] Step 2: Executing pep8 and pyflakes Tests (flake8)."
echo
flake8 simplebayes tests
echo "Exit Code:" $?

echo
echo " [Cahoots] Step 3: Executing pylint Tests"
echo
pylint simplebayes tests --reports=no
echo "Exit Code:" $?
echo
