#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from .._common.base import get_monkey_patching_base_class
from .metaaddress import MetaAddress
from .reference import Reference

_substitutions = {
    "Reference": Reference,
    "MetaAddress": MetaAddress,
}

MonkeyPatchingBaseClass = get_monkey_patching_base_class(
    _substitutions,
)
