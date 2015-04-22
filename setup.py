# coding: utf-8
from setuptools import setup

setup (
    name = 'simplebayes',
    version = '1.5.8',
    url = 'https://github.com/hickeroar/simplebayes',
    author = 'Ryan Vennell',
    author_email = 'ryan.vennell@gmail.com',
    description = 'A memory-based, optional-persistence naïve bayesian text classifier.',
    long_description = open('README.rst', 'r').read(),
    license = 'MIT',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Utilities',
    ],
    packages = ['simplebayes'],
)
