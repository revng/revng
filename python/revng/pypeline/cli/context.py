#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Concatenate, ParamSpec, Protocol, TypeVar
from typing import override

import click

# Import these only when type-checking to avoid circular imports
if TYPE_CHECKING:
    from revng.pypeline.cli.wrappers import Wrapper
    from revng.pypeline.pipeline import Pipeline


# Placeholder value for an uninitialized field of `ContextObject`
_MISSING = object()


@dataclass(kw_only=True)
class ContextObject:
    # The base directory, this is the directory where all relative path-based
    # lookup should use
    base_directory: Path
    # The path of the cache directory to use
    cache_dir: str
    # The pipebox module object
    pipebox: Any
    # Args that will be passed to the pipebox.initialize function
    pipebox_args: list[str]
    # Path of the python module loaded as pipebox
    pipebox_path: Path
    # Pipeline object, loaded from the yaml
    pipeline: "Pipeline"
    # Path to the pipeline yaml that has been loaded
    pipeline_path: Path
    # URL of the storage provider in use
    storage_provider_url: str
    # Optional Wrapper object, e.g. if the user specified `--gdb`.
    # This might cause the current process to be re-executed, during the
    # second execution this will be set back to None.
    wrapper: "Wrapper | None" = None
    # If set to true, help messages will also report help for artifacts that
    # have a category with `show_by_default == False`
    show_hidden_artifacts: bool = False

    # If we had to be 100% dataclass compliant all of the fields in this
    # dataclass should be `| None = None` because when the dataclass is
    # created, when the CLI start, these values have yet to be parsed. However
    # this makes things a lot less ergonomic since these values will eventually
    # be populated. It's also annoying and confusing to sprinkle
    # `assert X is None` throughout the codebase for values that will never be
    # effectively `None`. To alleviate this problem the following solution has
    # been implemented:
    # * Have a `make` method that stuffs placeholder values when the instance
    #   of this class is created at the start of the CLI
    # * Override `__getattribute__` so that if we accidentally try and access a
    #   placeholder value we get an exception
    @classmethod
    def make(cls, **kwargs) -> "ContextObject":
        # Create the dataclass following the usual dataclass logic, except that
        # fields that do not have a default will get the placeholder `_MISSING`
        for field_metadata in fields(cls):
            if field_metadata.name in kwargs:
                continue

            if field_metadata.default is not dataclasses.MISSING:
                kwargs[field_metadata.name] = field_metadata.default
            elif field_metadata.default_factory is not dataclasses.MISSING:
                kwargs[field_metadata.name] = field_metadata.default_factory()
            else:
                kwargs[field_metadata.name] = _MISSING
        return cls(**kwargs)

    # Since we cheat, we override attribute access so that if we actually mess
    # up and forget to assign a value we get an assertion
    @override
    def __getattribute__(self, name: str, /) -> Any:
        obj = object.__getattribute__(self, name)
        if obj is _MISSING:
            raise AttributeError(f"tried to access undefined field {name}")
        return obj


class _ObjProtocol(Protocol):
    obj: ContextObject


class ClickContext(_ObjProtocol, click.Context):
    pass


P = ParamSpec("P")
R = TypeVar("R")


def pass_context(f: Callable[Concatenate[ClickContext, P], R]) -> Callable[P, R]:
    return click.pass_context(f)  # type: ignore
