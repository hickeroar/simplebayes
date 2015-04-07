#!/usr/bin/env bash

python3 ./setup.py sdist upload
python3 ./setup.py bdist_egg upload
python ./setup.py bdist_egg upload
python3 ./setup.py bdist_wheel --universal upload
