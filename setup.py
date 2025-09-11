from setuptools import find_packages, setup
from typing import List

def get_req(filepath: str):
    """
    Returns the list of requirements
    """
    requirements = []
    with open(filepath) as fileobj:
        requirements = fileobj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="saumya",
    author_email="saumyapathak358@gmail.com",
    packages=find_packages(),  # Look for packages in current directory
    install_requires=get_req("requirements.txt"),
)