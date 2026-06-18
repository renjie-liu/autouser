"""Allow `python -m autouser` as an alias for the `autouser` console script."""

import sys

from autouser.cli import main

sys.exit(main())
