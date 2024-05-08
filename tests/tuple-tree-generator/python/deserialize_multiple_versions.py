#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

import yaml
from testmodule.v1 import RootType as RootV1
from testmodule.v1 import YamlLoader as YamlLoaderV1
from testmodule.v2 import RootType as RootV2
from testmodule.v2 import YamlLoader as YamlLoaderV2


def assert_parsing_fails(file, loader):
    try:
        yaml.load(file, Loader=loader)
    except Exception:
        return

    raise Exception("Parsing did not fail")


def test_deserialize_multiple_versions():
    """Tests that the custom YAML loaders can be used to deserialize multiple conflicting versions
    at the same time. Also tests that deserializing using an invalid version fails.
    """
    with open("v1.yml", encoding="utf-8") as f:
        v1_serialized = yaml.load(f, Loader=YamlLoaderV1)
        print(f"Type: {type(v1_serialized)}")
        assert isinstance(v1_serialized, RootV1)
    with open("v2.yml", encoding="utf-8") as f:
        v2_serialized = yaml.load(f, Loader=YamlLoaderV2)
        assert isinstance(v2_serialized, RootV2)

    with open("v1.yml", encoding="utf-8") as f:
        assert_parsing_fails(f, YamlLoaderV2)
    with open("v2.yml", encoding="utf-8") as f:
        assert_parsing_fails(f, YamlLoaderV1)

    print("test_deserialize_multiple_versions: OK")


test_deserialize_multiple_versions()
