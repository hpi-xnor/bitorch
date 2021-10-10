from pathlib import Path

import setuptools

root_path = Path(__file__).resolve().parent

requirements = [requirement.strip() for requirement in (root_path / 'requirements.txt').open().readlines()]
requirements_dev = [requirement.strip() for requirement in (root_path / 'requirements-dev.txt').open().readlines()]
requirements_optional = [
    requirement.strip() for requirement in (root_path / 'requirements-optional.txt').open().readlines()
]
print("dev:", requirements_dev, "opt:", requirements_optional)

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
        "opt": requirements_optional,
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
