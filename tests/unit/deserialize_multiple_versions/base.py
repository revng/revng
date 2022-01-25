from .._common.base import get_monkey_patching_base_class

_substitutions = {}

MonkeyPatchingBaseClass = get_monkey_patching_base_class(
    _substitutions,
    register_global_yaml_helpers=False,
)
YamlLoader = MonkeyPatchingBaseClass.YamlLoader
YamlDumper = MonkeyPatchingBaseClass.YamlDumper
