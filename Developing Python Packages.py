''' From Loose Code to Local Package '''

'''
Starting a package
Why build a package anyway?
- To make your code easier to reuse
- To avoid lots of coping and pasting
- To keep your functions up to date
- To give your code to others

Scripts, modules and packages
- Script - A Python file which is run like python myscript.py
- Package - A directory full of Python code to be imported
* e.g numpy
- Subpackage - A smaller package inside a package
* e.g numpy.random and numpy.linalg
- Module - A Python file inside a package which stores the package code.
- Library - Either a package, or a collection of packages.
* e.g the Python standard library (math, os, datetime, ...)

Directory tree of a package
Directory tree for simple package

mysimplepackage/
|--  simplemodule.py
|--  __init__.py 

* This directory, called mysimplepackage, is a Python Package
* simplemodule.py contains all the package code
* __init__.py marks this directory as a Python package.

Contents of simple package
__init__.py 

* Empty file


simplemodule.py 

def cool_function():
    ...
    return cool_result

...
def another_cool_function():
    ...
    return another_cool_result

* File with generalized functions and code.

Subpackages
Directory tree for package with subpackages

mysklearn/
|-- __init__.py
|-- preprocessing
|    |-- __init__.py
|    |-- normalize.py
|    |-- standardize.py
|-- regression
|    |-- __init__.py
|    |-- regression.py
|-- utils.py
'''


def count_words(filepath, words_list):
    # Open the text file
    with open(filepath) as file:
        text = file.read()

    n = 0
    for word in text.split():
        # Count the number of times the words in the list appear
        if word.lower() in words_list:
            n += 1

    return n


'''
Documentation
Why include documentation?
- Helps your users use your code
- Document each
* Function
* Class
* Class method

help() function
- It is used to view documentations

import numpy as np
help(np.sum) 

help(np.array) 

x = np.array([1, 2, 3, 4])
help(x.mean)

Function documentation
def count_words(filepath, words_list):
    """Count the total number of times these words appear.

    The count is performed on a text file at the given location.

    [explain what filepath and words_list are]

    [what is returned]
    """

Documentation style
- Google documentation style
"""Summary line.

Extended description of function.

Args:
    arg1 (int): Description of arg1
    arg2 (str): Description of arg2


- NumPy style
    """Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1 ...


-  reStructured text style
    """Summary line.

    Extended description of function.

    :Param arg1: Description of arg1
    :type arg1: int
    :Param arg2: Description of arg2
    :type arg2: str


- Epytext style
"""Summary line.

  Extended description of function.

  @type arg1: int
  @Param arg1: Description of arg1
  @type arg2: str
  @Param arg2: Description of arg2

NumPy documentation style
Popular in scientific Python packages like
- numpy
- scipy
- pandas
- sklearn
- matplotlib
- dask
- etc.

Other sections
- Raises
- See Also
- Notes
- References
- Examples

Documentation templates and style translation
- pyment can be used to generaate docstrings
- Run from terminal *
- Any documentation style from
* Google
* Numpydoc
* reST (i.e. reStructured-text)
* Javadoc (i.e epytext)
- Modify documentation from one style to another

pyment -w -o numpydoc textanalysis.py

def count_words(filepath, words_list):
    # Open the text file
    ...
    return n   -> in

* -w : overwrite file
* -o numpydoc : output in NumPy style

def count_words(filepath, words_list):
    """

    Parameters
    ----------
    filepath :

    words_list :


    Returns
    -------
    type
    """  -> out

Translate to Google style
pyment -w -o google textanalysis.py

def count_words(filepath, words_list):
    """Count the total number of times these words appear.

    The count is performed on a text file at the given location.

    Parameters
    ----------
    filepath : str
        Path to text file
    words_list : list of str
        Count the total number of appearances of these words.

    Returns
    -------
-> in

def count_words(filepath, words_list):
    """Count the total number of times these words appear.

    The count is performed on a text file at the given location.

    Args:
        filepath(str): Path to text file
        words_list(list of str): Count the total number of appearances of these words.

    Returns

    """  -> out

Package, subpackage and module documentation
mysklearn/__init__.py 

"""
Linear regression for Python
==================
mysklearn is a complete package for implementing linear regression in python.


mysklearn/preprocessing/__init__.py 

"""
A subpackage for standard preprocessing operations.
"""


mysklearn/preprocessing/normalize.py 

"""
A module for normalizing data.
"""
'''


'''
Structuring imports
- Directory tree for package with subpackages

mysklearn/
|-- __init__.py
|-- preprocessing
|    |-- __init__.py
|    |-- normalize.py
|    |-- standardize.py
|-- regression
|    |-- __init__.py
|    |-- regression.py
`-- utils.py


Without package imports
import mysklearn
help(mysklearn)

import mysklearn.preprocessing
help(mysklearn.preprocessing)

import mysklearn.preprocessing.normalize
help(mysklearn.preprocessing.normalize)


Importing subpackages into packages
mysklearn/__init__.py

Absolute import
from mysklearn import preprocessing

* Used most - more explicit

Relative import
from . import preprocessing

* Used sometimes - shorter and sometimes simpler


Importing modules
We imported preprocessing into mysklearn

import mysklearn
help(mysklearn.preprocessing)

But preprocsessing has no link to normalize

import mysklearn
help(mysklearn.preprocessing.normalize)

mysklearn/preprocessing/__init__.py

Absolute import
from mysklearn.preprocessing import normalize

Relative import
from . import normalize


Restructuring imports
import mysklearn

help(mysklearn.preprocessing.normalize.normalize_data)

Import function into subpackage
mysklearn/preprocessing/__init__.py

Absolute import
from mysklearn.preprocessing.normalize import normalize_data

Relative import
from .normalize import normalize_data

import mysklearn

help(mysklearn.preprocessing.normalize_data)


Importing between sibling modules
- Directory tree for package with subpackages

mysklearn/
|-- __init__.py
|-- preprocessing
|    |-- __init__.py
|    |-- normalize.py  <--
|    |-- funcs.py
|    |-- standardize.py
|-- regression
|    |-- __init__.py
|    |-- regression.py
`-- utils.py

In normalize.py

Absolute import
from mysklearn.preprocessing.funcs import (mymax, mymin)

Relative import
from .funcs import mymax, mymin


Importing between modules far apart
- Directory tree for package with subpackages

mysklearn/
|-- __init__.py
|-- preprocessing
|    |-- __init__.py
|    |-- normalize.py  <--
|    |-- standardize.py  <--
|-- regression
|    |-- __init__.py
|    |-- regression.py
`-- utils.py

- A custom exception MyException is in utils.py

- In normalize.py, standardize.py and regression.py

Absolute import
from mysklearn.utils import MyException

Relative import
from ..utils import MyException


Relative import cheat sheet
- from . import module
* From current directory, import module
- from .. import module
* Frome one directory up, import module
- from .module import function
* From module in current directory, import function
- from ..subpackage.module import function
* From subpackage one directory up, from module in that subpackage, import function
'''


''' Install Your Package from Anywhere '''

'''
Installing your own package
Why should you install your own package?
Directory tree for package with subpackages

home/
|-- mysklearn/      <-- in same directory
|  |-- __init__.py
|  |-- preprocessing
|  |    |-- __init__.py
|  |    |-- normalize.py 
|  |    |-- standardize.py
|  |-- regression
|  |    |-- __init__.py
|  |    |-- regression.py
|  `-- utils.py -> out
|-- example_script.py     <-- in same directory

Inside example_script.py

import mysklearn

Directory tree 

home/
|-- mypackages
|    |-- mysklearn/     <-- 
|        |-- __init__.py
|        |-- preprocessing
|        |    |-- __init__.py
|        |    |-- normalize.py 
|        |    |-- standardize.py
|        |-- regression
|             |-- __init__.py
|             |-- regression.py
`-- myscripts
    `-- example_script.py     <-- 

Inside example_script.py

import mysklearn    X

setup.py
- Is used to install the package
- Contains metadata on the package

Package directory structure
Directory tree for package with subpackages

mysklearn/      <-- outer directory
|-- mysklearn/      <-- innner source code directory
|    |-- __init__.py
|    |-- preprocessing
|    |    |-- __init__.py
|    |    |-- normalize.py 
|    |    |-- standardize.py
|    |-- regression
|    |    |-- __init__.py
|    |    |-- regression.py
|    |-- utils.py 
|-- setup.py    <-- setup script in outer

Inside setup.py
# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author='James Fulton',
    description='A complete package for linear regression.',
    name='mysklearn',
    version='0.1.0',
    packages=find_packages(include=['mysklearn', 'mysklearn.*'])
)

version number = (major number).(minor number).(patch number)

Editable installation
pip install -e .

* . = package in current directory
* -e = editable
'''


'''
Dealing with dependencies
What are dependencies?
- Other packages you import inside your package
- Inside mymodule.py:

# These imported packages are dependencies
import numpy as np
import pandas as pd
...

Adding dependencies to setup.py
from setuptools import setup, find_packages

setup(
    ...
    install_requires=['pandas', 'scipy', 'matplotlib'],
)

Controlling dependency version
from setuptools import setup, find_packages

setup(
    ...
    install_requires=[
        'pandas>=1.0',              # good
        'scipy==1.1',               # bad
        'matplotlib>=2.2.1,<3'      # good
    ],
)

- Allow as many package versions as possible
- Get rid of unused dependencies

Python versions
from setuptools import setup, find_packages

setup(
    ...
    python_requires='>=2.7, !=3.0.*, !=3.1.*',
)

Choosing dependency and package versions
- Check the package history or release notes
* e.g the NumPy release notes
- Test different versions

Making an environment for developers
Save package requirements to a file

pip freeze > requirements.txt

mysklearn/
|-- mysklearn
|    |-- __init__.py
|    |-- preprocessing
|    |    |-- __init__.py
|    |    |-- normalize.py 
|    |    |-- standardize.py
|    |-- regression
|    |    |-- __init__.py
|    |    |-- regression.py
|    |-- utils.py 
|-- setup.py   
|-- requirements.txt    <-- developer environment

Install requirements from file

pip install -r requirements.txt
'''


'''
Including licences and writing READMEs
Why do i need aa license?
- To give others permission to use your code

Open source licenses
- Find more information https://choosealicense.com
- Allow users to
* use your package
* modify your package
* distribute versions of your package

What is a README?
- The 'front page' of your package
- Displayed on Github or PyPI

What to include in a README
README sections
- Title
- Description and Features
- Installation
- Usage examples
- Contributing
- License

README format
Markdown (commonmark)
- Contained in README.md file
- Simpler
- Used in this course and in the wild

reStructuredText
- Contained in README.rst file
- More complex
- Also common in the wild


Commonmark
Contents of README.md

# mysklearn
mysklearn is a package for complete **linear regression** in Python.

You can find out more about this package on [DataCamp](https://datacamp.com)

## Installation
You can instaall this package using

```
pip install mysklearn
``` 


What it looks like when rendered

mysklearn
mysklearn is a package for complete linear regression in Python.

You can find out more about this package on DataCamp

Installation
You can instaall this package using

pip install mysklearn


Adding these files to your package
Directory tree for package with subpackages

mysklearn/
|-- mysklearn
|    |-- __init__.py
|    |-- preprocessing
|    |    |-- ...
|    |-- regression
|    |    |-- ...
|    |-- utils.py 
|-- setup.py   
|-- requirements.txt
|-- LICENSE  <--- new files
|-- README.md  <--- added to top directory


MANIFEST.in
Lists all the extra files to include in your package distribution.

Contents of MANIFEST.in

include LICENSE
include README.md

mysklearn/
|-- mysklearn
|    |-- __init__.py
|    |-- preprocessing
|    |    |-- ...
|    |-- regression
|    |    |-- ...
|    |-- utils.py 
|-- setup.py   
|-- requirements.txt
|-- LICENSE  
|-- README.md
|-- MANIFEST.in  <--- 
'''


'''
Publishing your package
PyPI
Python Package Index
- pip installs packages from here
- Anyone can upload packages
- You should upload your package as soon as it might be useful

Distributions
- Distribution package - a bundled version of your package which is ready to install
- Source distribution - a distribution packaage which is mostly your source code
- Wheel distribution -  a distribution package which has been processed to make it faster to install

How to build distributions
python setup.py sdist bdist_wheel

- sdist = source distribution
- bdist_wheel = wheel distribution

mysklearn/
|-- mysklearn
|-- setup.py   
|-- requirements.txt
|-- LICENSE  
|-- README.md
|-- dist  <--- 
|    |-- mysklearn-0.1.0-py3-none-any.whl
|    |-- mysklearn-0.1.0.tar.gz
|-- build
|-- mysklearn.egg-info


Getting your package out there
Upload your distributions to PyPI

twine upload dist/*

Upload your distributions to TestPI

twine upload -r testpypi dist/*


How other people can install your package
Install package from PyPI

pip install mysklearn

Install package from TestPyPI

pip install --index-url    https://test.pypi.org/simple
            --extra-index-url    htpps://pypi.org/simple
            mysklearn
'''


''' Increasing Your Package Quality '''

'''
Testing your package
The art and disccipline of testing
Imagine you are working on this function

def get_ends(x):
    """Get the first and last element in a list"""
    return x[0], x[-1]

You might test it to make sure it works

# Check the function
get_ends([1, 1, 5, 39, 0]) -> in

(1, 0) -> out

Good packages brag about how many tests they have 
* 91% of the pandas package code has tests

Writing tests
def get_ends(x):
    """Get the first and last element in a list"""
    return x[0], x[-1]

def test_get_ends():
    assert get_ends([1, 5, 39, 0]) == (1, 0)
    assert get_ends(['n', 'e', 'r', 'd']) == ('n', 'd')

test_get_ends()

Organizing tests inside your package
mysklearn/
|-- mysklearn  <-- package
|-- tests <- tests directory
|-- setup.py
|-- LICENSE
|-- MANIFEST.in

Organizing tests inside your package
Test directory layout

mysklearn/tests/
|-- __init__.py
|-- preprocessing
|    |-- __init__.py
|    |-- test_normalize.py 
|    |-- test_standardize.py
|-- regression
|    |-- __init__.py
|    |-- test_regression.py
|-- test_utils.py 

Code directory layout

mysklearn/mysklearn/
|-- __init__.py
|-- preprocessing
|    |-- __init__.py
|    |-- normalize.py 
|    |-- standardize.py
|-- regression
|    |-- __init__.py
|    |-- regression.py
|-- utils.py 

Organizing a test module
Inside test_normalize.py

from mysklearn.preprocessing.normalize import ( find_max, find_min, normalize_data )

def test_find_max(x):
    assert find_max([1, 4, 7, 1]) == 7

def test_find_min(x):
    assert ...

def test_normalize_data(x):
    assert ...


Inside normalize.py

def find_max(x):
    ...
    return x_max

def find_min(x):
    ...
    return x_min

def normalize_data(x):
    ...
    return x_norm


Running tests with pytest
mysklearn/ <-- navigate to here
|-- mysklearn  
|-- tests 
|-- setup.py
|-- LICENSE
|-- MANIFEST.in

pytest  -> in

* pytest looks inside the test directory
* It looks for modules like test_modulename.py
* It looks for functions like test_functionname()
* It runs these functions and shows output

======================== test session starts ========================
platform linux -- Python 3.7.9, pytest-6.1.2, py-1.9.0, pluggy-0.13.1
rootdir: /home/workspace/mypackages/mysklearn
collected 6 items

tests/preprocessing/test_normalize.py ...                       [ 50%]
tests/preprocessing/test_standardize.py ...                     [100%]

======================== 6 passed in 0.23s =========================== -> out
'''


'''
Testing your package with different environments
Testing multiple versions of Python 
This setup.py allows any version of Python from version 2.7 upwards.

from setuptools import setup, find_packages

setup(
    ...
    python_requires='>=2.7',
)

To test these Python versions you must:
- Install all these Python versions
- Run tox

What is tox?
- Designed to run tests with multiple versions of Python

Configure tox
Configuration file - tox.ini

mysklearn/
|-- mysklearn
|    |-- ...
|-- tests
|    |-- ...
|-- setup.py
|-- LICENSE
|-- MANIFEST.in
|-- tox.ini <--- configuration file

[tox]
envlist = py27, py35, py36, py37

[testenv]
deps = pytest
commands = 
    pytest
    echo 'run more commands'
    ...

* Headings are surrounded by square brackets [...]
* To test Python version X.Y add pyXY to envlist
* The versions of Python you test need to be installed already
* The commands parameter lists the terminal commands tox will run
* The commands list can be any commands which will run from the terminal, like ls, cd, echo etc.

Running tox
tox

mysklearn/      <-- navigate to here
|-- mysklearn
|    |-- ...
|-- tests
|    |-- ...
|-- setup.py
|-- LICENSE
|-- MANIFEST.in
|-- tox.ini 

tox output
py27 create: /mypackages/mysklearn/.tox/py27
py27 installdeps: pytest
py27 inst: /mypackages/mysklearn/.tox/.tmp/package/1/mysklearn-0.1.0.zip
py27 installed: mysklearn==0.1.0,numpy==1.16.6,pandas==0.24.2,pytest==4.6.11, ...
py27 run-test-pre: PYTHONHASHSEED='2837498672'
...

py27 run-test: commands[0] | pytest
======================== test session starts ========================
platform linux2 -- Python 2.7.17, ...
rootdir: /home/workspace/mypackages/mysklearn
collected 6 items

tests/preprocessing/test_normalize.py ...                       [ 50%]
tests/preprocessing/test_standardize.py ...                     [100%]

======================== 6 passed in 0.23s =========================== 

...
_____________________________ summary ________________________________
    py27: commands succeeded
    py35: commands succeeded
    py36: commands succeeded
    py37: commands succeeded    -> out
'''


'''
Keeping your package stylish
Introducing flake8
- Standard Python style is described in PEP8
- A style guide dictates how code should be laid out
- pytest is used to find bugs
- flake8 is used to find styling mistakes

Running flake8
Static code checker - reads code but doesn't run

flake8 features.py -> in

features.py:2:1: f401 'math' imported but unused
.. -> out

<filename>:<line number>:<character number>:<error code> <description>

Breaking the rules on pupose
quadratic.py
quadratic_1 = 6 * x**2 + 2 * x + 4;     # noqa (no quality assurance)
quadratic_2 = 12 * x**2 + 2 * x + 8     # noqa : E222

flake8 settings
Ignoring style violations without using comments
flake8 --ignore E222 quadratic.py -> in

quadratic.py:5:35: E703 statement ends with a semicolon  -> in

flake8 --select F401, F841 features.py -> in

2:1: F401 'math' imported but unused
9:5: F841 local variable 'mean_x' is assigned to but never used  -> out

Choosing package settings using setup.cfg
Package file tree

.
|-- example_package
|    |-- __init__.py
|    `-- example_package.py
|-- tests
|    |-- __init__.py
|    `-- test_example_package.py
|-- README.rst
|-- LICENSE
|-- MANIFEST.in
|-- setup.py
|-- setup.cfg

Create a setup.cfg to store settings
[flake8]

ignore = E302
exclude = setup.py

per-file-ignores = example_package/example_package.py: E222

The whole package
$ flake8

Use the least filtering possible
Least filtering
1.  # noqa : <code>
2.  # no qa
3.  setup.py -> per-file-ignores
4.  setup.py -> exclude, ignore
Most filtering
'''
"""Main module."""


''' Rapid Package Development '''

'''
Faster package development with templates
Package file tree
.
|-- example_package
|    |-- __init__.py
|    `-- example_package.py
|-- tests
|    |-- __init__.py
|    `-- test_example_package.py
|-- README.rst  <-
|-- LICENSE  <-- lots
|-- MANIFEST.in  <-- of 
|-- tox.in  <-- additional
|-- setup.py  <-- files
|-- setup.cfg  <--

Templates
- Python packages have lots of extra files
- There is a lot to remember
- Using templates takes care of a lot of this

cookiecutter
- Can be used to create empty Python packages
- Creates all the additional files your package needs

Using cookiecutter
cookiecutter <template-url>

cookiecutter https://github.com/audreyr/cookiecutter-pypackage -> in

fullname [Audrey Roy Greenfeld]: James Fulton
email [audreyr@example.com]: james@email.com
github_username [audreyr]: MyUsername
project_name [Python Boilerplate]: mysklearn
project_slug [mysklearn]: mysklearn -> out
...
project_short_description [Python Boilerplate ...]: A Python package for linear regression.
pypi_username [MyUsername]:
version [0.1.0]:
...
use_pytest [n]: y
use_pypi_deployment_with_travis [y] : n
add_pyup_badge [n]: n
...
Select command_line_interface:
1 - Click
2 - Argparse
3 - No command-line interface
Choose frome 1, 2, 3 [1]: 3
create_author_file [y]: y
...
Select open_source_license:
1 - MIT license
2 - BSD license
3 - ISC license
4 - Apache Software License 2.0
5 - GNU General Public License v3
6 - Not open source
Choose from 1, 2, 3, 4, 5, 6 [1]: 6 -> out

- Project slug - the name used in pip install name


Template output

mysklearn/
|-- mysklearn/
|    |-- __init__.py
|    `-- mysklearn.py
|-- tests/
|    |-- __init__.py
|    `-- test_mysklearn.py
|-- MANIFEST.in 
|-- README.rst
|-- requirements_dev.txt
|-- setup.cfg
|-- setup.py
|-- tox.in 
|-- AUTHORS.rst
|-- CONTRIBUTING.rst
|-- HISTORY.rst
`-- Makefile
...
...
|-- docs/
|-- .github/
|-- .editorconfig
|-- .gitignore
`-- .travis.yml

Inside the AUTHORS.rst file

=====
Credits
=====

Development Lead
---------------------

* JAmes Fulton <james@example.com>

Contributors
--------------

None yet. Why not be  the first?
'''


'''
Version numbers and history
Final files
* CONTRIBUTING.md
* HISTORY.md

CONTRIBUTING.md
- Either markdown or reStructured-Text
- Invites other developers to work on your package
- Tells them how to get started

HISTORY.md
e.g NumPy release notes
- Known as history, changelog or release notes
- Tells users what has changed between versions

- Section for each released version
- Bullet points of the important changes
- Subsections for 
* Improvements to existing functions
* New additions
* Bugs that have been fixed
* Deprecations

# History

## 0.30
### Changed
- Regression fitting sped up using NumPy operations.
### Deprecated
- Support for Python 3.5 has ended
- `regression.regression` module has been removed

## 0.2.1
### Fixed
- Fixed bug causing intercepts of zero

## 0.2.0
### Added
- Multiple linear regression now available in new `regression.multiple_regression` module.
### Deprecated
- 0.2.x will be the last version that supports Python 3.5
- `regression.regression` module has been renamed `regression.single_regression`. `regression.regression` will be removed in next minor release. -> in

Version number
- Increase version number when ready for new release
- Cannot upload to PyPI if not changed

mysklearn/
|-- mysklearn/
|    |-- __init__.py  <---
|    `-- mysklean.py
|-- setup.py  <---
...

The package version number
setup.py

# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    ...
    version='0.1.0', <---
    ...

)

Top level __init__.py

"""
Linear regression for Python
==================

mysklearn is a complete package for implementing linear regression in python.
"""

__version__ = '0.1.0'  <---

bumpversion
- Convenient tool to update all package version numbers

mysklearn/  <-- navigate to here
|-- mysklearn/
|    |-- __init__.py  <---
|    `-- mysklean.py
|-- setup.py  <---
...

bumpversion major

bumpversion minor

bumpversion patch
'''


'''
Makefiles and classifiers
Classifiers
- Metadata for your package
- Helps users find your package on PyPI
- You should include
* Package status
* Your intended audience
* License type
* Language
* Versions of Python supported
- Lots more classifiers exist (https://pypi.org/classifiers)

Inside setup.py of mysklearn

setup(
    ...
    classifiers=[
        'Development Status : : 2 - Pre-Alpha',
        'Intended Audience : : Developers',        
        'License : : OSI Approved : : MIT License',
        'Natural Language : : English',
        'Programming Language : : Python : : 3',
        'Programming Language : : Python : : 3.6',
        'Programming Language : : Python : : 3.7',
        'Programming Language : : Python : : 3.8',
    ],
    ...
)

What are Makefiles for?
- Used to automate parts of building your package

mysklearn/
...
|-- README.md
|-- setup.py
|-- Makefile  <---
...

What is in a Makefile?
Inside Makefile
...

dist: ## builds source and wheel package 
    python3 setup.py sdist bdist_wheel

clean-build: ## remove build artifacts
    rm -fr build/
    rm -fr dist/
    rm -fr .eggs/

test: ## run tests quickly with the default Python
    pytest

release: dist ## package and upload a release
    twine upload dist/*

How do i use the Makefile?
make <function-name>

mysklearn/    <--- navigate to here
...
|-- README.md
|-- setup.py
|-- Makefile
...

To use the dist function type this in terminal

make dist

Makefile summary
- list the functions in a make file and what they do

make help -> in
'''
