# Bitorch

## Usage

Similar to recent versions of [torchvision](https://github.com/pytorch/vision), you should be using Python 3.8 or newer.

Install the requirements and add this path to your _PYTHONPATH_:
```bash
pip install -r requirements.txt
# add to PYTHONPATH with virtualenvwrapper:
add2virtualenv . 
```

## Development

Install the development requirements with:
```bash
 pip install -r requirements-dev.txt
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
