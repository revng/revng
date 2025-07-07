#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Temporary workaround until all the pypeline python classes have defined methods
# mypy: disable-error-code="attr-defined"

import os
import sys
from collections.abc import Buffer
from ctypes import CDLL

import yaml

from revng.pypeline import Analysis, Container, Model, ObjectID, Pipe, get_registry


def load_extension(paths: list[str]):
    """Load extension modules via ctypes.CDLL (equivalent to `dlopen`)"""
    handles = []

    for path in paths:
        # RTLD_GLOBAL is needed due to nanobind using weak symbols to derive the TypeID
        handles.append(CDLL(path, os.RTLD_NOW | os.RTLD_GLOBAL))

    import revng.internal._cpp_pypeline as ext

    return (ext, handles)


def check_names(ext):
    """Check that _cpp_pypeline has all the classes we expect it to have"""

    # Names that we know are always present in `_cpp_pypeline`
    known_names = ("__init_revng", "Buffer")

    names = [
        x for x in dir(ext) if not ((x.startswith("__") and x.endswith("__")) or x in known_names)
    ]

    # Check that the classes in the registry all came from the ext module
    for type_ in (Analysis, Container, Model, ObjectID, Pipe):
        registry = get_registry(type_)
        for key, value in registry.items():
            assert issubclass(value, type_)
            assert key in names
            assert getattr(ext, key) is value
            names.remove(key)

    # Check that all the classes have been registered, the names variable
    # should be empty at this point
    assert len(names) == 0


def compare_dicts(dict1: dict[ObjectID, Buffer], dict2: dict[ObjectID, Buffer]):
    assert all(isinstance(k, ObjectID) and isinstance(v, Buffer) for k, v in dict1.items())
    assert all(isinstance(k, ObjectID) and isinstance(v, Buffer) for k, v in dict2.items())
    assert dict1.keys() == dict2.keys()
    for key in dict1.keys():
        assert memoryview(dict1[key]) == memoryview(dict2[key])


def check_pipeline():
    """Check that all the components of fakePipebox behave as expected. This
    function **should not** use the `_cpp_pypeline` module directly, rather
    rely on the `get_registy` functionality to retrieve the classes"""

    # Create a new model
    model = get_registry(Model)["Model"]()
    assert isinstance(model, Model)

    # Check that the model is cloneable
    model2 = model.clone()
    assert isinstance(model2, Model)

    # Create a couple of ObjectIDs
    foo = ObjectID.make("/function/0x400000:Code_x86_64")
    bar = ObjectID.make("/function/0x400020:Code_x86_64")
    assert isinstance(foo, ObjectID)
    assert isinstance(bar, ObjectID)
    # Check equality works
    assert foo == ObjectID.make("/function/0x400000:Code_x86_64")

    # Create a StringContainer and deserialize some data to it
    container = get_registry(Container)["StringContainer"]()
    container.deserialize({foo: b"foo", bar: b"bar"})
    container_serialized = container.serialize({foo})
    compare_dicts(container_serialized, {foo: b"foo"})
    # Re-feed the serialized data back to the container, this is to make sure
    # that the `Buffer` C++ class is also handled by `deserialize`
    container.deserialize(container_serialized)

    # Run the AppendFooPipe
    pipe = get_registry(Pipe)["AppendFooPipe"]("{}")  # type: ignore[call-arg]
    pipe.run(model, [container], [[]], [[foo]], "")
    container_serialized2 = container.serialize({foo, bar})
    compare_dicts(container_serialized2, {foo: b"foofoo", bar: b"bar"})

    # Run the AppendFooLibAnalysis
    analysis = get_registry(Analysis)["AppendFooLibAnalysis"]()
    analysis.run(model, [container], [[foo]], "")

    model_serialized = model.serialize()
    assert isinstance(model_serialized, Buffer)
    yaml_model = yaml.safe_load(bytes(model_serialized))
    assert "foo.so" in yaml_model["ImportedLibraries"]


def main():
    ext, _ = load_extension(sys.argv[1:])
    check_names(ext)
    check_pipeline()


if __name__ == "__main__":
    main()
