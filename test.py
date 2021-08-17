
from argparse import ArgumentParser


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument('-1', '--flag-1', action='store_true', default=False)
    parser.add_argument('-2', '--flag-2', action='store_true', default=False)
    parser.add_argument('-3', '--flag-3', action='store_true', default=False)

    args, unknown = parser.parse_known_args()
    print(f"args        : {args}")
    print(f"unknown     : {unknown}")

    hidden = ArgumentParser(add_help=False)
    hidden.add_argument('-d', '--debug', action='store_true', default=False)
    hidden_args = hidden.parse_args(unknown)
    print(f"hidden_args : {hidden_args}")


if __name__ == "__main__":
    _parse_args()
