from pathlib import Path
from typing import Union

import setuptools

root_path = Path(__file__).resolve().parent


def get_requirements(file_path: Union[Path, str]):
    return [requirement.strip() for requirement in (root_path / file_path).open().readlines()]


requirements = get_requirements('requirements.txt')
requirements_dev = get_requirements('requirements-dev.txt')

setuptools.setup(
    name="bitorch",
    version="0.1.0",
    author="Joseph Bethge",
    author_email="joseph.bethge@hpi.de",
    description="A package for building and training quantized and binary neural networks with Pytorch",
    packages=setuptools.find_packages(exclude='tests'),
    install_requires=requirements,
    extras_require={
        "dev": requirements_dev,
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.8',
)
