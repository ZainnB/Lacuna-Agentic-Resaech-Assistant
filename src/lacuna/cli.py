"""`lacuna` command line. Each package registers its own subcommands via `register(subparsers)`.

Packages that are not built yet are skipped, so the CLI works while the tree is incomplete.
"""

import argparse
import importlib

SUBCOMMAND_MODULES = [
    "lacuna.db",
    "lacuna.llm.cli",
    "lacuna.ingest.cli",
    "lacuna.graph.cli",
    "lacuna.scout.cli",
    "lacuna.papers.cli",
    "lacuna.extract.cli",
    "lacuna.mcp_server.cli",
    "lacuna.evals.cli",
    "lacuna.deepread.cli",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lacuna", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in SUBCOMMAND_MODULES:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as e:
            # A lacuna package that isn't built yet, imported directly or by a dependency, hides
            # only the subcommands that need it. A missing third-party module is a real error.
            if e.name and (name.startswith(e.name) or e.name.startswith("lacuna.")):
                continue
            raise
        module.register(sub)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
