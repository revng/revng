#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Temporary workaround until all the pypeline python classes have defined methods
# mypy: disable-error-code="attr-defined"

import os
import sys
from ctypes import CDLL

import yaml

from revng.internal.pypeline import Analysis, Container, Model, ObjectID, Pipe, get_registry

handles = []

for param in sys.argv[1:]:
    # RTLD_GLOBAL is needed due to nanobind using weak symbols to derive the TypeID
    handles.append(CDLL(param, os.RTLD_NOW | os.RTLD_GLOBAL))

import revng.internal._cpp_pypeline as ext  # noqa: E402

KNOWN_NAMES = ("__init_revng", "Buffer")

names = [x for x in dir(ext) if not ((x.startswith("__") and x.endswith("__")) or x in KNOWN_NAMES)]

# Check that the classes in the registry all came from the ext module
for type_ in (Analysis, Container, Model, ObjectID, Pipe):
    registry = get_registry(type_)
    for key, value in registry.items():
        assert issubclass(value, type_)
        assert key in names
        assert getattr(ext, key) is value
        names.remove(key)

# Check that all the classes have been registered, the names variable should be
# empty at this point
assert len(names) == 0

# We no longer need ext, this avoids code below this line from using the
# classes directly and instead using the `get_registry` function
del ext


def to_bytes(in_: dict):
    return {k: bytes(v) for k, v in in_.items()}


# Create a new model
model = get_registry(Model)["Model"]()
assert isinstance(model, Model)

# Check that the model is cloneable
model2 = model.clone()

# Create a couple of ObjectIDs
foo = ObjectID.make("/function/0x400000:Code_x86_64")
bar = ObjectID.make("/function/0x400020:Code_x86_64")
# Check equality works
assert foo == ObjectID.make("/function/0x400000:Code_x86_64")

# Create a StringContainer and deserialize some data to it
container = get_registry(Container)["StringContainer"]()
container.deserialize({foo: b"foo", bar: b"bar"})
assert to_bytes(container.serialize({foo})) == {foo: b"foo"}

# Run the AppendFooPipe
pipe = get_registry(Pipe)["AppendFooPipe"]("{}")  # type: ignore[call-arg]
pipe.run(model, [container], [[]], [[foo]], "")
assert to_bytes(container.serialize({foo, bar})) == {foo: b"foofoo", bar: b"bar"}

# Run the AppendFooLibAnalysis
analysis = get_registry(Analysis)["AppendFooLibAnalysis"]()
analysis.run(model, [container], [[foo]], "")

yaml_model = yaml.safe_load(bytes(model.serialize()))
assert "foo.so" in yaml_model["ImportedLibraries"]
