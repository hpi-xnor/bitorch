from pathlib import Path
from typing import Union, List

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


def _get_requirements(*file_path: Union[Path, str]):
    requirements_list = []
    for fp in file_path:
        with (root_path / fp).open() as requirements_file:
            requirements_list.extend(list(requirement.strip() for requirement in requirements_file.readlines()))
    # exclude bitorch from examples
    if "bitorch" in requirements_list:
        requirements_list.remove("bitorch")
    return requirements_list


def _get_files_recursively(glob: str, root: str = ".") -> List[str]:
    return list(str(x) for x in Path(root).rglob(glob))


with open("README.md", "r", encoding="utf-8") as handle:
    readme_content = handle.read()


setuptools.setup(
    name="bitorch",
    url="https://github.com/hpi-xnor/bitorch",
    version=version,
    author="Hasso Plattner Institute",
    author_email="fb10-xnor@hpi.de",
    description="A package for building and training quantized and binary neural networks with Pytorch",
    long_description=readme_content,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=_get_requirements("requirements.txt"),
    extras_require={
        "dev": _get_requirements("requirements-dev.txt"),
        # "opt": _get_requirements(*_get_files_recursively("requirements*.txt", root="examples")),
        "opt": _get_requirements("examples/image_classification/requirements.txt", "examples/mnist/requirements.txt"),
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.7",
    data_files=[
        (
            ".",
            [
                "AUTHORS",
                "CHANGELOG.md",
                "mypy.ini",
                "version.txt",
            ]
            + _get_files_recursively("requirements*.txt")
            + _get_files_recursively("README.md", root="examples"),
        ),
    ],
)
