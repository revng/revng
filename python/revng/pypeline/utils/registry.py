#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# The singleton registry of subclasses for all classes decorated with `registry`
_REGISTRY_SINGLETON: dict[type, dict[str, type]] = {}


def get_registry[T](cls: type[T]) -> dict[str, type[T]]:
    """
    Get the registry of subclasses for a given class which was decorated with
    `registry`.
    """
    if cls not in _REGISTRY_SINGLETON:
        raise TypeError(f'"{cls}" is not a registered class')
    # Clone to avoid mutation
    return dict(_REGISTRY_SINGLETON[cls])


def get_singleton[T](cls: type[T]) -> type[T]:
    """
    Get the singleton subclass for a given class which was decorated with
    `registry`.
    """
    ty_registry = get_registry(cls)
    if len(ty_registry) != 1:
        raise ValueError(
            f"Expected exactly one singleton for {cls}, but found "
            f"{len(ty_registry)}: {list(ty_registry.keys())}"
        )
    # Get the single value from the registry
    return next(iter(ty_registry.values()))


def register_all_subclasses(cls: type, *, singleton: bool = False) -> None:
    """
    Register all subclasses of the given class `cls` in the global registry.
    """
    if cls not in _REGISTRY_SINGLETON:
        _REGISTRY_SINGLETON[cls] = {}

    def find_leafs(c: type) -> list[type]:
        """
        Recursively find all leaf subclasses of a given class `c`.
        """
        subclasses = c.__subclasses__()
        if not subclasses:
            return [c]
        leafs = []
        for subclass in subclasses:
            leafs.extend(find_leafs(subclass))
        return leafs

    for leaf in find_leafs(cls):
        _REGISTRY_SINGLETON[cls][leaf.__name__] = leaf

    if singleton and len(_REGISTRY_SINGLETON[cls]) > 1:
        raise ValueError(
            f"Expected exactly one singleton for {cls}, but found "
            f"{len(_REGISTRY_SINGLETON[cls])}: {list(_REGISTRY_SINGLETON[cls].keys())}"
        )
