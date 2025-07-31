#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# It's important that the `cli` submodule is NOT imported here
# as for the cli to work we need to first load the pipebox and then
# load the cli submodules so they can access the custom types.

__version__ = "@VERSION@"
