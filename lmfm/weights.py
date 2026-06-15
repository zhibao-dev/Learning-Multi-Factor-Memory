"""Load a learned weights.json back into a MemoryValue."""

from __future__ import annotations

import json
from pathlib import Path

from .value import default_memory_value


def load_value_from_weights_file(path):
    data = json.loads(Path(path).read_text())
    weights = data.get("weights", data)   # accept bare {factor: w} too
    return default_memory_value(override=weights)
