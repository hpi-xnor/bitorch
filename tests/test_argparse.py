import pytest


def test_argparse():
    arg_parser = pytest.importorskip("examples.pytorch_lightning.utils.arg_parser")
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        _ = arg_parser.create_argparser(["main.py", "-h"])
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0
