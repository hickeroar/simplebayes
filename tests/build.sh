#!/bin/bash
set -e

echo
echo " [simplebayes] Step 1: Executing Unit Tests"
echo
pytest tests/ --cov=simplebayes --cov-fail-under=100 -v
rm -f .coverage*

echo
echo " [simplebayes] Step 2: Executing flake8"
echo
flake8 simplebayes tests

echo
echo " [simplebayes] Step 3: Executing pylint"
echo
pylint simplebayes tests --exit-zero
echo
