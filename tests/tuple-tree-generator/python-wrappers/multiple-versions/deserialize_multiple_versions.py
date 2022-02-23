#!/usr/bin/env python3

import yaml
from testmodule.v1._generated import RootType as RootV1
from testmodule.v2._generated import RootType as RootV2
from testmodule.v1.base import YamlLoader as YamlLoaderV1
from testmodule.v2.base import YamlLoader as YamlLoaderV2


def assert_parsing_fails(serialized, root_type):
    try:
        root_type.parse_obj(serialized)
    except:
        return

    raise Exception("Parsing did not fail")


def test_deserialize_multiple_versions():
    """Tests that the custom YAML loaders can be used to deserialize multiple conflicting versions at the same time.
    Also tests that deserializing using an invalid version fails.
    """
    with open("v1.yml") as f:
        v1_serialized = yaml.load(f, Loader=YamlLoaderV1)
        print(f"Type: {type(v1_serialized)}")
        assert type(v1_serialized) is RootV1
    with open("v2.yml") as f:
        v2_serialized = yaml.load(f, Loader=YamlLoaderV2)
        assert type(v2_serialized) is RootV2

    RootV1.parse_obj(v1_serialized)
    RootV2.parse_obj(v2_serialized)
    assert_parsing_fails(v1_serialized, RootV2)
    assert_parsing_fails(v2_serialized, RootV1)

    print("test_deserialize_multiple_versions: OK")


def test_tagged_deserialize_multiple_versions():
    """Tests that the custom YAML loaders can be used to deserialize multiple conflicting versions at the same time.
    Differing from test_deserialize_multiple_versions, this test uses tagged documents to test that the default path
    resolver installed in the `YamlLoader`s does not cause issues.
    Also tests that deserializing using an invalid version fails.
    """
    with open("v1_tagged.yml") as f:
        v1_serialized = yaml.load(f, Loader=YamlLoaderV1)
        assert type(v1_serialized) is RootV1
    with open("v2_tagged.yml") as f:
        v2_serialized = yaml.load(f, Loader=YamlLoaderV2)
        assert type(v2_serialized) is RootV2

    with open("v1_tagged.yml") as f:
        assert_parsing_fails(f, YamlLoaderV2)
    with open("v2_tagged.yml") as f:
        assert_parsing_fails(f, YamlLoaderV1)

    print("test_tagged_deserialize_multiple_versions: OK")


test_deserialize_multiple_versions()
test_tagged_deserialize_multiple_versions()
