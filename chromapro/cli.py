"""CLI entrypoint."""

from .migrate import run_cli


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
