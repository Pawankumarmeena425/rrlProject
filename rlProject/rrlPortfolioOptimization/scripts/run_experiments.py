import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rrl_portfolio.experiment import run_full_study


def main() -> None:
    table = run_full_study()
    print(table)


if __name__ == "__main__":
    main()
