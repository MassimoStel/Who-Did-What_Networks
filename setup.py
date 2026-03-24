from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get the requirements from the requirements.txt file
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="teanets",
    url="https://github.com/RiccardoImprota/Agent-did-Target-Networks",
    author="Riccardo Improta",
    author_email="riccardo.imp@gmail.com",
    packages=find_packages(),
    install_requires=requirements,
    version="0.2.5",
    license="BSD-3-Clause license",
    description="An example of a python package from pre-existing code",
)
