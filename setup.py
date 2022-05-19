import subprocess
from pathlib import Path
from typing import Union

import setuptools

root_path = Path(__file__).resolve().parent

version = "unknown"
version_file = root_path / "version.txt"
if version_file.exists():
    with open(root_path / "version.txt") as handle:
        version_content = handle.read().strip()
        if version_content:
            version = version_content
print("version:", version)


def get_requirements(file_path: Union[Path, str]):
    return [requirement.strip() for requirement in (root_path / file_path).open().readlines()]


with open("README.md", "r", encoding="utf-8") as handle:
    readme_content = handle.read()

setuptools.setup(
    name="bitorch",
    url="https://github.com/hpi-xnor/bitorch",
    version=version,
    author="Joseph Bethge",
    author_email="joseph.bethge@hpi.de",
    description="A package for building and training quantized and binary neural networks with Pytorch",
    long_description=readme_content,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude='tests'),
    install_requires=get_requirements('requirements.txt'),
    extras_require={
        "dev": get_requirements('requirements-dev.txt'),
        "opt": get_requirements('requirements-opt.txt'),
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires='>=3.7',
    data_files=[
        ('.', [
            'version.txt',
            'requirements.txt',
            'requirements-dev.txt',
            'requirements-opt.txt',
        ]),
    ]
)
