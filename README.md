# BITorch

BITorch is a library currently under development to simplify building quantized and binary neural networks
with [PyTorch](https://pytorch.org/).
This is an early preview version of the library.
If you wish to use it and encounter any problems, please create an issue.
Our current roadmap contains:

- Extending the model zoo with pre-trained models of state-of-the-art approaches
- Adding examples for advanced training methods with multiple stages, knowledge distillation, etc.

All changes are tracked in the [changelog](CHANGELOG.md).

## Installation

Similar to recent versions of [torchvision](https://github.com/pytorch/vision), you should be using Python 3.8 or newer.
Currently, the only supported installation is pip (a conda package is planned in the future).

### Pip

If you wish to use a *specific version* of PyTorch for compatibility with certain devices or CUDA versions,
we advise on installing the corresponding versions of `pytorch` and `torchvision` first (or afterwards),
please consult [pytorch's getting started guide](https://pytorch.org/get-started/locally/).

Afterwards simply run:
```bash
pip install bitorch
```

Note, that you can also request a specific PyTorch version directly, e.g. for CUDA 11.3:
```bash
pip install bitorch --extra-index-url https://download.pytorch.org/whl/cu113
```

To use advanced logging capabilities with [tensorboardX](https://github.com/lanpa/tensorboardX),
install the optional dependencies as well:

```bash
pip install "bitorch[opt]"
```

#### Local and Development Install Options

The package can also be installed locally for editing and development.
First, clone the [repository](https://github.com/hpi-xnor/bitorch), then run:

```bash
pip install -e .
```

To activate advanced logging with Tensorboard and model summary, install the optional dependencies as well:

```bash
pip install -e ".[opt]"
```

### Dali Preprocessing

If you want to use the [Nvidia dali preprocessing library](https://github.com/NVIDIA/DALI),
e.g. with CUDA 11.x, (currently only supported for imagenet)
you need to install the `nvidia-dali-cuda110` package by running the following command:

```
 pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
```

### Code formatting and typing

Install the _dev_ requirements for (local) development:

```bash
pip install -e ".[dev]"
```

New code should be compatible with Python 3.X versions and be compliant with PEP8. To check the codebase, please run

```bash
flake8
```

The codebase has type annotations, please make sure to add type hints if required. We use `mypy` for type checking:

```bash
mypy --config-file mypy.ini
```

Finally, the tests can be run with:

```bash
pytest
```
