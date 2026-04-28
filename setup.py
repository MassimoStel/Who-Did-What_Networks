from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file (optional)
try:
    with open(path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

# Get the requirements from the requirements.txt file (optional)
try:
    with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = []

setup(
    name="teanets",
    url="https://github.com/MassimoStel/TEA_Networks.git",
    author="Sebastiano Franchini",
    author_email="franchini.sebastiano@gmail.com",
    packages=find_packages(),
    install_requires=requirements,
    version="0.3.0",
    license="BSD-3-Clause license",
    description="Target-Event-Agent Networks: SVO extraction and analysis from text",
)

