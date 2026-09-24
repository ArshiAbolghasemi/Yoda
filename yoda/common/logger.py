"""Shared project loggers.

Two streams, because they have different audiences:

``logger``
    Diagnostics - what the run is doing, how long it took, what it decided.
    Goes to **stderr**, timestamped, so it interleaves with progress bars and
    stays out of anything the caller redirects.

``results``
    The program's actual output - metric tables, report paths. Goes to
    **stdout**, unadorned, so ``main.py experiments > results.txt`` captures
    exactly the results and none of the diagnostics.

Nothing in this project uses ``print``: a table written with ``print`` cannot
be silenced, levelled or redirected independently of the diagnostics.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("yoda")

results = logging.getLogger("yoda.results")
results.setLevel(logging.INFO)
# Bare format: this stream is read by people and by `>`, not parsed as logs.
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
results.addHandler(_handler)
results.propagate = False  # never duplicated into the diagnostic stream
