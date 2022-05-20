# BITorch

BITorch is a library currently under development to simplify building quantized and binary neural networks
with [PyTorch](https://pytorch.org/).
This is an early preview version of the library.
If you wish to use it and encounter any problems, please create an issue.
Our current roadmap contains:

- Extending the model zoo with pre-trained models of state-of-the-art approaches
- Adding examples for advanced training methods with multiple stages, knowledge distillation, etc.

All changes are tracked in the [changelog](https://github.com/hpi-xnor/bitorch/blob/main/CHANGELOG.md).

Please refer to [our wiki](https://bitorch.readthedocs.io/en/latest/) for a comprehensive introduction into
the library or use the introduction notebook in `examples/notebooks`.

## Installation

Similar to recent versions of [torchvision](https://github.com/pytorch/vision), you should be using Python 3.8 or newer.
Currently, the only supported installation is pip (a conda package is planned in the future).

### Pip

If you wish to use a _specific version_ of PyTorch for compatibility with certain devices or CUDA versions,
we advise on installing the corresponding versions of `pytorch` and `torchvision` first (or afterwards),
please consult [pytorch's getting started guide](https://pytorch.org/get-started/locally/).

Otherwise, simply run:
```bash
pip install bitorch
```

Note, that you can also request a specific PyTorch version directly, e.g. for CUDA 11.3:
```bash
pip install bitorch --extra-index-url https://download.pytorch.org/whl/cu113
```

If you want to run the examples install the optional dependencies as well:
```bash
pip install "bitorch[opt]"
```

#### Local and Development Install Options

The package can also be installed locally for editing and development.
First, clone the [repository](https://github.com/hpi-xnor/bitorch), then run:

```bash
pip install -e .         # without optional dependencies
pip install -e ".[opt]"  # with optional dependencies
```

### Dali Preprocessing

If you want to use the [Nvidia dali preprocessing library](https://github.com/NVIDIA/DALI),
e.g. with CUDA 11.x, (currently only supported for imagenet)
you need to install the `nvidia-dali-cuda110` package by running the following command:

```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
```

## Development

Install the package and _dev_ requirements locally for development:

```bash
pip install -e ".[dev]"
```

### Tests

The tests can be run with [pytest](https://docs.pytest.org/):

```bash
pytest
```

### Code formatting and typing

For conveniently checking whether your code suites the required style (more details below), run
```bash
./check-codestyle.sh
```

New code should be compatible with Python 3.X versions and be compliant with PEP8. To check the codebase, please run
```bash
flake8
```

The codebase has type annotations, please make sure to add type hints if required. We use `mypy` for type checking:
```bash
mypy --config-file mypy.ini
```

For code formatting we use `black`:
```bash
black . --check --verbose --diff --color  # check what changes the formatter would do
black .  # apply the formatter
```

In order to automatically apply the code formatting with every commit, you can also install pre-commit
and use the pre-commit hook:
```bash
pre-commit install
```

### Documentation

We use [Google's Python Docstring Format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
to document our code.

Documentation can be generated with
```bash
sphinx-build -b html docs/source/ docs/build/ -a
```
