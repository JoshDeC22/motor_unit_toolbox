from setuptools import setup, find_packages

from __init__ import __version__

def fetch_requirements():
    with open("requirements.txt", "r", encoding="utf-8", errors="ignore") as f:
        reqs = f.read().strip().split("\n")
    return reqs

setup(
    name='motor_unit_toolbox',
    version=__version__,
    author='Irene Mendez Guerra',
    author_email='irene.mendez17@imperial.ac.uk',
    description='motor_unit_toolbox is a package to analyse motor unit activity.',

    url='https://github.com/imendezguerra/motor_unit_toolbox',

    packages=find_packages(),
    install_requires=fetch_requirements(),
)
