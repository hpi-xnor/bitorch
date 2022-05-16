.. bitorch documentation installation file, created by
   sphinx-quickstart on Fri Apr  8 13:58:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Build Docs
===================================

The docs for :code:`bitorch` are generated using the `sphinx <https://www.sphinx-doc.org/en/master/>`_ package.
To build the docs, `cd` into the repository root and execute.

.. code-block:: bash

    sphinx-build -b html docs/source/ docs/build/ -a

The generated :code:`.rst` files will be put into :code:`docs/build`.