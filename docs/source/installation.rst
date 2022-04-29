.. bitorch documentation installation file, created by
   sphinx-quickstart on Fri Apr  8 13:58:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
===================================

Similar to recent versions of `torchvision <https://github.com/pytorch/vision>`_, you should be using Python 3.8 or newer.

Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you wish to use a specific version of PyTorch for compatibility with certain devices or CUDA versions,
we advise on installing the corresponding versions of `pytorch` and `torchvision` first,
please consult `pytorch's getting started guide <https://pytorch.org/get-started/locally/>`_.
A good solution to use CUDA 11.x is to install the packages :code:`torch==1.9.0+cu111` and :code:`torchvision==0.10.0+cu111` first.

Install the package with pip (the :code:`--find-links` option can be removed if torch and torchvision have already been installed):

.. code-block:: bash

    pip install bitorch --find-links https://download.pytorch.org/whl/torch_stable.html

To use advanced logging capabilities with `tensorboardX <https://github.com/lanpa/tensorboardX>`_, install the optional dependencies as well:

.. code-block:: bash

    pip install "bitorch[opt]" --find-links https://download.pytorch.org/whl/torch_stable.html


Local and Development Install Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package can also be installed locally for editing and development.
First, clone the `repository <https://github.com/hpi-xnor/bitorch>`_, then run:

.. code-block:: bash

    pip install -e . --find-links https://download.pytorch.org/whl/torch_stable.html


To activate advanced logging with Tensorboard and model summary, install the optional dependencies as well:

.. code-block:: bash

    pip install -e ".[opt]" --find-links https://download.pytorch.org/whl/torch_stable.html

Make sure the *dev* option is used for (local) development:

.. code-block:: bash

    pip install -e ".[dev]" --find-links https://download.pytorch.org/whl/torch_stable.html

Dali Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use the Nvidia dali preprocessing library (currently only supported for imagenet) you need to install the :code:`nvidia-dali-cuda110` package by running the following command:

.. code-block:: bash

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

Code formatting and typing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New code should be compatible with Python 3.X versions and be compliant with PEP8. To check the codebase, please run

.. code-block:: bash

    flake8 --config=setup.cfg .

The codebase has type annotations, please make sure to add type hints if required. We use :code:`mypy` for type checking:

.. code-block:: bash

    mypy --config-file mypy.ini

Finally, the tests can be run with:

.. code-block:: bash

    pytest
