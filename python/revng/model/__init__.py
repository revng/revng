#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Automatically import the latest version
from glob import glob
from pathlib import Path


def _get_most_recent_version():
    most_recent_version = 0
    search_pattern = str(Path(__file__).parent / "v*")
    for dirpath in glob(search_pattern):
        dirname = Path(dirpath).name
        version = int(dirname[1:])
        if version > most_recent_version:
            most_recent_version = version

    return f"v{most_recent_version}"


_latest_version = _get_most_recent_version()

# Equivalent to `from .<latest_version> import *`
_module = __import__(
    _latest_version,
    globals=globals(),
    locals=locals(),
    fromlist=("*",),
    level=1,  # Perform a relative import
)

if hasattr(_module, "__all__"):
    all_names = _module.__all__
else:
    all_names = [name for name in dir(_module) if not name.startswith("_")]
globals().update({name: getattr(_module, name) for name in all_names})
