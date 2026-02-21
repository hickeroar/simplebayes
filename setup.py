# coding: utf-8
from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='simplebayes',
    version='3.0.0',
    url='https://github.com/hickeroar/simplebayes',
    author='Ryan Vennell',
    author_email='ryan.vennell@gmail.com',
    description=(
        'A memory-based, optional-persistence naïve bayesian text classifier.'
    ),
    long_description=long_description,
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Utilities',
    ],
    python_requires='>=3.10',
    install_requires=[
        'snowballstemmer>=3.0.1',
        'fastapi>=0.116.1',
        'uvicorn[standard]>=0.35.0',
    ],
    packages=find_packages(include=['simplebayes', 'simplebayes.*']),
    package_data={'simplebayes': ['py.typed']},
    entry_points={
        'console_scripts': ['simplebayes-server=simplebayes.cli:run'],
    },
)
