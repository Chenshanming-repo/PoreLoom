#!/usr/bin/env python3
"""Compatibility entry point for the current frozen publication build."""
from build_revision import main

if __name__ == '__main__':
    print('This entry point now rebuilds the 4 November 2026 revision. '
          'Historical scripts are in paper/legacy/20261030/scripts/.')
    main()
