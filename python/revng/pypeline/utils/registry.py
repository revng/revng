#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import inspect
from typing import Callable, Optional, overload

# The singleton registry of subclasses for all classes decorated with `registry`.
_REGISTRY_SINGLETON: dict[type, dict[str, type]] = {}


def get_registry(cls: type) -> dict[str, type]:
    """
    Get the registry of subclasses for a given class which was decorated with
    `registry`.
    """
    if cls not in _REGISTRY_SINGLETON:
        raise TypeError("`cls` is not a registered class")
    return dict(_REGISTRY_SINGLETON[cls])  # Clone to avoid mutation


def get_singleton(cls: type) -> type:
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
    return next(iter(ty_registry.values()))  # Get the single value from the registry


@overload
def registry(cls: type, /) -> type: ...


@overload
def registry(*, singleton: bool = False) -> Callable[[type], type]: ...


def registry(
    cls: Optional[type] = None, /, *, singleton: bool = False
) -> type | Callable[[type], type]:
    """Class decorator that registers all subclasses of the decorated class.
    The set of subclasses can be retrieved using the `get_registry` function.
    If `singleton` is set to True, only one subclass can be registered.

    Example usage:
    ```python
    from revng.pypeline.utils import registry, get_registry

    @registry
    class MyBaseClass:
        pass

    class SubClassA(MyBaseClass):
        pass
    class SubClassB(MyBaseClass):
        pass

    assert get_registry(MyBaseClass) == {
        "SubClassA": SubClassA,
        "SubClassB": SubClassB,
    }
    ```
    """
    if cls is not None and not isinstance(cls, type):
        raise TypeError(f"The `cls` decorated with @registry must be a class, not {cls}")

    def wrap(cls) -> type:
        _REGISTRY_SINGLETON[cls] = {}
        # Extract the original __init_subclass__ method so we can support decorating
        # classes that already have a custom __init_subclass__ method.
        original_init_subclass = cls.__init_subclass__

        def __init_subclass__(cls_param, *args, **kwargs) -> None:  # noqa: N807
            # The builtin __init_subclass__ raises an error if the method is called with
            # any arguments.
            if not inspect.isbuiltin(original_init_subclass):
                original_init_subclass.__func__(cls_param, *args, **kwargs)
            assert issubclass(
                cls_param, cls
            ), f"Invalid container subclass `{cls_param}` is not of a subclass of `{cls}`"
            if singleton:
                assert len(_REGISTRY_SINGLETON[cls]) == 0, (
                    f"Container `{cls_param}` is a singleton and cannot be "
                    f"registered with other containers {_REGISTRY_SINGLETON[cls]}"
                )
            _REGISTRY_SINGLETON[cls][cls_param.__name__] = cls_param

        cls.__init_subclass__ = classmethod(__init_subclass__)
        return cls

    # This is needed to make it work both as @registry and @registry(singleton=True)
    if cls is None:
        return wrap
    return wrap(cls)
