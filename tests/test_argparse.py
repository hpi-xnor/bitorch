import pytest
from argparse import ArgumentParser


# this test checks for naming conflicts by adding all arguments to one parser
def test_argparse():
    arg_parser = pytest.importorskip("examples.pytorch_lightning.utils.arg_parser")
    parser = arg_parser.create_argparser()
    # parser.parse_args(['main.py', '-h'])
