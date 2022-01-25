from .._common.base import get_monkey_patching_base_class

_substitutions = {}

_classnames_to_tags = {}

MonkeyPatchingBaseClass = get_monkey_patching_base_class(
    _substitutions,
    classnames_to_tags=_classnames_to_tags,
    register_global_yaml_helpers=False,
)
YamlLoader = MonkeyPatchingBaseClass.YamlLoader
YamlDumper = MonkeyPatchingBaseClass.YamlDumper
