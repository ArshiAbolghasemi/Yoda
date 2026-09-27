"""The torch device, resolved from the hardware the process is running on.

There is no setting. CUDA if the box has it, Apple MPS if it is a Mac with a
usable Metal backend, CPU otherwise - decided once and reused, so the gate and
the RL controller cannot end up on different devices in the same run.

What this does *not* mean is that the project needs a GPU. Only two components
are torch at all (the attention control gate and the SAC controller), and both
are dominated by work no accelerator touches: the gate's targets come from
convex solves, and every RL step is another one. Picking up an accelerator when
one is present is the least surprising behaviour, not a performance claim.
"""

from __future__ import annotations

from functools import cache

import torch

from yoda.common.logger import logger


@cache
def resolve_device() -> torch.device:
    """``cuda`` > ``mps`` > ``cpu``, whichever this machine actually has."""
    if torch.cuda.is_available():
        name = "cuda"
        detail = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        name, detail = "mps", "apple metal"
    else:
        name, detail = "cpu", "no accelerator"
    logger.info("device_resolved device=%s (%s)", name, detail)
    return torch.device(name)
