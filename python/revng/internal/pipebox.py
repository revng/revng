#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.internal.support import import_pipebox
from revng.support import get_root

_module, _handles = import_pipebox([get_root() / "lib/librevngPipebox.so"])
