from argparse import ArgumentParser
from examples.pytorch_lightning.utils.arg_parser import add_regular_args, add_all_model_args

# this test checks for naming conflicts by adding all arguments to one parser


def test_argparse():
    parser = ArgumentParser()
    add_regular_args(parser)
    add_all_model_args(parser)
