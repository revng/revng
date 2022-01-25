from .._common.base import get_monkey_patching_base_class
from .metaaddress import MetaAddress
from .typeref import Typeref

_substitutions = {
    "Typeref": Typeref,
    "MetaAddress": MetaAddress,
}

_classnames_to_tags = {
    "EnumModel": "Enum",
    "UnionModel": "Union",
}

MonkeyPatchingBaseClass = get_monkey_patching_base_class(
    _substitutions,
    classnames_to_tags=_classnames_to_tags,
    register_global_yaml_helpers=False,
)
YamlLoader = MonkeyPatchingBaseClass.YamlLoader
YamlDumper = MonkeyPatchingBaseClass.YamlDumper
