#!/usr/bin/env python3
"""Command-line interface for NYISO data processing."""

from dispatch_benchmark.io.nyiso import main as nyiso_main


def main():
    """Entry point for the NYISO CLI command."""
    nyiso_main()


if __name__ == "__main__":
    main()
