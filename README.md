# Bitorch

## Usage

Similar to recent versions of [torchvision](https://github.com/pytorch/vision), you should be using Python 3.8 or newer.

Install the package with pip:
```bash
pip install -e .
```

To activate advanced logging with Tensorboard and model summary, install the optional dependencies as well:
```bash
pip install -e ".[opt]"
```

## Development

Install the _dev_ package with:
```bash
 pip install -e .[dev]
```
Use quotes, i.e., `".[dev]"` in zsh.

### Code formatting and typing

New code should be compatible with Python 3.X versions and be compliant with PEP8. To check the codebase, please run
```bash
flake8 --config=setup.cfg .
```

The codebase has type annotations, please make sure to add type hints if required. We use `mypy` tool for type checking:
```bash
mypy --config-file mypy.ini
```
