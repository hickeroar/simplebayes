# coding: utf-8
import setuptools
from distutils.core import setup

setup (
    name = 'simplebayes',
    version = '1.2.0',
    url = 'https://github.com/hickeroar/simplebayes',
    author = 'Ryan Vennell',
    author_email = 'ryan.vennell@gmail.com',
    description = 'A memory-based, optional-persistence naïve bayesian text classifier.',
    long_description = open('README.rst', 'r').read(),
    license = open('LICENSE', 'r').read(),
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Utilities',
    ],
    packages = ['simplebayes'],
)
# python ./setup.py sdist upload
