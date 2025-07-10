#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from ctypes import CDLL

from revng.support import get_root

handle = CDLL(str((get_root() / "lib/libfakePipebox.so").resolve()), os.RTLD_NOW | os.RTLD_LOCAL)

import revng.internal.my_ext as ext  # noqa: E402

model = ext.Model()
foo = ext.ObjectID()
foo.deserialize("/function/0x400000:Code_x86_64")

bar = ext.ObjectID()
bar.deserialize("/function/0x400020:Code_x86_64")

a = ext.StringContainer()
a.deserialize({foo: b"foo", bar: b"bar"})
print(a.serialize({foo}))


b = ext.AppendFooPipe()
b.run(model, [a], [[]], [[foo]], "")
print({k.serialize(): v for k, v in a.serialize({foo, bar}).items()})


c = ext.PrintFooAnalysis()
c.run(model, [a], [[foo]], "")
