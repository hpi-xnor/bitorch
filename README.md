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

### Pip

If you wish to use a specific version of PyTorch for compatibility with certain devices or CUDA versions,
we advise on installing the corresponding versions of `pytorch` and `torchvision` first,
please consult [pytorch's getting started guide](https://pytorch.org/get-started/locally/).
A good solution to use CUDA 11.x is to install the packages `"torch==1.9.0+cu111" "torchvision==0.10.0+cu111"` first.

Install the package with pip (the `--find-links` option can be removed if torch and torchvision have already been installed):
```bash
pip install bitorch --find-links https://download.pytorch.org/whl/torch_stable.html
```

To use advanced logging capabilities with [tensorboardX](https://github.com/lanpa/tensorboardX), install the optional dependencies as well:
```bash
pip install "bitorch[opt]" --find-links https://download.pytorch.org/whl/torch_stable.html
```

#### Local and Development Install Options

The package can also be installed locally for editing and development.
First, clone the [repository](https://github.com/hpi-xnor/bitorch), then run:
```bash
pip install -e . --find-links https://download.pytorch.org/whl/torch_stable.html
```

To activate advanced logging with Tensorboard and model summary, install the optional dependencies as well:
```bash
pip install -e ".[opt]" --find-links https://download.pytorch.org/whl/torch_stable.html
```

Make sure the _dev_ option is used for (local) development:
```bash
pip install -e ".[dev]" --find-links https://download.pytorch.org/whl/torch_stable.html
```

### Code formatting and typing

New code should be compatible with Python 3.X versions and be compliant with PEP8. To check the codebase, please run
```bash
flake8 --config=setup.cfg .
```

The codebase has type annotations, please make sure to add type hints if required. We use `mypy` for type checking:
```bash
mypy --config-file mypy.ini
```

Finally, the tests can be run with:
```bash
pytest
```
