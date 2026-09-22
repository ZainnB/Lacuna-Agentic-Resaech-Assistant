"""Run the full eval and write eval/results/<date>_<config-hash>.{json,md}.

Equivalent to `figrag eval`; extra flags are passed through (--limit, --ids, --no-judge).
"""
import sys

from figrag.cli import main

if __name__ == "__main__":
    main(["eval", *sys.argv[1:]])
