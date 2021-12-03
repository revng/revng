from .._common.base import get_monkey_patching_base_class
from .typeref import Typeref
from .metaaddress import MetaAddress

_substitutions = {
    "Typeref": Typeref,
    "MetaAddress": MetaAddress,
}

_classnames_to_tags = {
    "EnumModel": "Enum",
    "UnionModel": "Union",
}

MonkeyPatchingBaseClass = get_monkey_patching_base_class(_substitutions, classnames_to_tags=_classnames_to_tags)
