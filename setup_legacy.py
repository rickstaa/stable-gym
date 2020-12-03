"""Setup file for the 'simzoo' python package.
"""

# Standard library imports
import logging
import os
from setuptools import setup, find_packages
import sys
from distutils.sysconfig import get_python_lib

# Get the relative path for including (data) files with the package
relative_site_packages = get_python_lib().split(sys.prefix + os.sep)[1]
date_files_relative_path = os.path.join(relative_site_packages, "simzoo")

# Additional python requirements that could not be specified in the package.xml
requirements = ["gym", "matplotlib"]

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#################################################
# Setup script ##################################
#################################################


# Parse readme.md
with open("README.md") as f:
    readme = f.read()

# Run python setup
setup(
    # name="simzoo",
    # setup_requires=["setuptools_scm"],
    use_scm_version=True,
    # description=("A python package containing several openai gym environments."),
    # long_description=readme,
    # long_description_content_type="text/markdown",
    # author="Rick Staa",
    # author_email="rick.staa@outlook.com",
    # license="Rick Staa copyright",
    # url="https://github.com/rickstaa/machine_learning_control",
    # keywords="rl, openai gym",
    # classifiers=[
    #     "Programming Language :: Python :: 3.5",
    #     "Programming Language :: Python :: 3.6",
    #     "Programming Language :: Python :: 3.7",
    #     "Natural Language :: English",
    #     "Topic :: Scientific/Engineering",
    # ],
    # install_requires=requirements,
    # extras_require={
    #     "dev": ["pytest", "bumpversion", "flake8", "black"],
    #     "build": ["gym==0.17.2", "matplotlib==3.3.1"],
    # },
    # packages=find_packages(),
    # include_package_data=True,
    # data_files=[(date_files_relative_path, ["README.md"])],
)
