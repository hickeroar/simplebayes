# coding: utf-8
import setuptools
from distutils.core import setup

setup (
    name = 'simplebayes',
    version = '1.3.1',
    url = 'http://hickeroar.github.io/simplebayes/',
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Utilities',
    ],
    packages = ['simplebayes'],
)
# python ./setup.py sdist upload
