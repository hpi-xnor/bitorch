# Bitorch

## Installation

Similar to recent versions of [torchvision](https://github.com/pytorch/vision), you should be using Python 3.8 or newer.

### Pip

- Install the package with pip (we need to add the `--find-links` option for torch/torchvision):
```bash
pip install bitorch --find-links https://download.pytorch.org/whl/torch_stable.html
```
- To use advanced logging capabilities, install the optional dependencies as well:
```bash
pip install -e "bitorch[opt]" --find-links https://download.pytorch.org/whl/torch_stable.html
```

#### Local and Development Install Options

The package can also be installed locally. First, clone this repository, then run:
```bash
pip install -e . --find-links https://download.pytorch.org/whl/torch_stable.html
```

- To activate advanced logging with Tensorboard and model summary, install the optional dependencies as well:
```bash
pip install -e ".[opt]" --find-links https://download.pytorch.org/whl/torch_stable.html
```
- Install the _dev_ package with:
```bash
 pip install -e ".[dev]" --find-links https://download.pytorch.org/whl/torch_stable.html
```

### Conda

Alternatively, you can install the following packages with conda:
```bash
conda install pytorch=1.9.0 torchvision=0.9.0 matplotlib
conda install flake8 mypy pytest # dev packages
```

### Code formatting and typing

New code should be compatible with Python 3.X versions and be compliant with PEP8. To check the codebase, please run
```bash
flake8 --config=setup.cfg .
```

The codebase has type annotations, please make sure to add type hints if required. We use `mypy` tool for type checking:
```bash
mypy --config-file mypy.ini
```
