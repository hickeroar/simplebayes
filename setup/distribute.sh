#!/usr/bin/env bash

# Build and upload to PyPI using twine (modern approach)
# Requires: pip install build twine

set -e

echo "Building package..."
python -m build

echo "Uploading to PyPI..."
twine upload dist/*
