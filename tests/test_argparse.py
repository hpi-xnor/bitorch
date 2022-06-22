import pytest
from argparse import ArgumentParser


# this test checks for naming conflicts by adding all arguments to one parser
def test_argparse():
    arg_parser = pytest.importorskip("examples.pytorch_lightning.utils.arg_parser")
    parser = ArgumentParser()
    arg_parser.add_regular_args(parser)
    arg_parser.add_all_model_args(parser)
