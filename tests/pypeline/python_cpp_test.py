#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys
from collections.abc import Buffer

import yaml

from revng.internal.support import import_pipebox
from revng.pypeline import initialize_pypeline
from revng.pypeline.analysis import Analysis, AnalysisBinding
from revng.pypeline.container import Container, ContainerDeclaration
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import Kind, ObjectID, ObjectSet
from revng.pypeline.pipeline import Pipeline
from revng.pypeline.pipeline_node import PipelineConfiguration, PipelineNode
from revng.pypeline.storage.memory import InMemoryStorageProvider
from revng.pypeline.storage.storage_provider import ContainerLocation
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.task.requests import Requests
from revng.pypeline.task.savepoint import SavePoint
from revng.pypeline.task.task import TaskArgument, TaskArgumentAccess
from revng.pypeline.utils.registry import get_registry, get_singleton


def check_names(ext):
    """Check that _pipebox has all the classes we expect it to have"""

    # Names that we know are always present in `_pipebox`
    known_names = ("Buffer",)

    names = [
        x for x in dir(ext) if not ((x.startswith("__") and x.endswith("__")) or x in known_names)
    ]

    # Check that the classes in the registry all came from the ext module
    for type_ in (Analysis, Container, Kind, Model, ObjectID, Pipe):
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
    function **should not** use the `_pipebox` module directly, rather
    rely on the `get_registy` functionality to retrieve the classes"""

    # Create a new model
    model = get_singleton(Model)()
    assert isinstance(model, Model)

    # Check that the model is cloneable
    model2 = model.clone()
    assert isinstance(model2, Model)

    objectid_cls = get_singleton(ObjectID)
    # Create a couple of ObjectIDs
    foo = objectid_cls.deserialize("/function/0x400000:Code_x86_64")
    bar = objectid_cls.deserialize("/function/0x400020:Code_x86_64")
    assert isinstance(foo, ObjectID)
    assert isinstance(bar, ObjectID)
    # Check equality works
    assert foo == objectid_cls.deserialize("/function/0x400000:Code_x86_64")

    # Create a StringContainer and deserialize some data to it
    StringContainer = get_registry(Container)["StringContainer"]  # noqa: N806
    container = StringContainer()
    container.deserialize({foo: b"foo", bar: b"bar"})
    container_serialized = container.serialize(ObjectSet.from_list([foo]))
    compare_dicts(container_serialized, {foo: b"foo"})
    # Re-feed the serialized data back to the container, this is to make sure
    # that the `Buffer` C++ class is also handled by `deserialize`
    container.deserialize(container_serialized)

    # Run the AppendFooPipe
    AppendFooPipe = get_registry(Pipe)["AppendFooPipe"]  # noqa: N806
    assert AppendFooPipe.signature() == (
        TaskArgument("Container", StringContainer, TaskArgumentAccess.READ_WRITE, ""),
    )
    pipe = AppendFooPipe("{}")
    pipe.run(
        model=ReadOnlyModel(model),
        containers=[container],
        incoming=[ObjectSet(foo.kind())],
        outgoing=[ObjectSet.from_list([foo])],
        configuration="",
    )
    container_serialized2 = container.serialize(ObjectSet.from_list([foo, bar]))
    compare_dicts(container_serialized2, {foo: b"foofoo", bar: b"bar"})

    # Run the AppendFooLibAnalysis
    AppendFooLibAnalysis = get_registry(Analysis)["AppendFooLibAnalysis"]  # noqa: N806
    assert AppendFooLibAnalysis.signature() == (StringContainer,)
    analysis = AppendFooLibAnalysis()
    analysis.run(
        model=model, containers=[container], incoming=[ObjectSet.from_list([foo])], configuration=""
    )

    model_serialized = model.serialize()
    assert isinstance(model_serialized, Buffer)
    yaml_model = yaml.safe_load(bytes(model_serialized))
    assert "foo.so" in yaml_model["ImportedLibraries"]


def check_simple_pipeline():
    StringContainer = get_registry(Container)["StringContainer"]  # noqa: N806
    AppendFooPipe = get_registry(Pipe)["AppendFooPipe"]  # noqa: N806
    AppendFooLibAnalysis = get_registry(Analysis)["AppendFooLibAnalysis"]  # noqa: N806

    model = get_singleton(Model)()
    storage_provider = InMemoryStorageProvider()
    child_cont = ContainerDeclaration("Container", StringContainer)
    declarations = [child_cont]
    pipeline_configuration: PipelineConfiguration = {}
    storage_provider.set_model(model.serialize())

    # Create the pipeline
    begin_node = PipelineNode(SavePoint("begin", to_save=declarations))
    inplace_node = PipelineNode(AppendFooPipe("{}"), bindings=[child_cont])
    end_node = PipelineNode(SavePoint("end", to_save=declarations))
    begin_node.add_successor(inplace_node)
    inplace_node.add_successor(end_node)
    pipeline = Pipeline(
        set(declarations),
        begin_node,
        analyses={AnalysisBinding(AppendFooLibAnalysis(), (child_cont,), begin_node)},
    )

    # Force the savepoint to store the objects
    container = StringContainer()

    objectid_cls = get_singleton(ObjectID)
    foo = objectid_cls.deserialize("/function/0x400000:Code_x86_64")
    bar = objectid_cls.deserialize("/function/0x400020:Code_x86_64")
    container.deserialize({foo: b"foo", bar: b"bar"})
    foo_bar_request = Requests({child_cont: ObjectSet.from_list([foo, bar])})

    # Pre-populate the storage
    begin_node.run(
        model=ReadOnlyModel(model),
        containers={child_cont: container},
        incoming=foo_bar_request,
        outgoing=foo_bar_request,
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )

    # Check
    begin_node_config = begin_node.configuration_id(pipeline_configuration)
    container_location_begin = ContainerLocation(1, "Container", begin_node_config)
    compare_dicts(storage_provider.storage[container_location_begin], {foo: b"foo", bar: b"bar"})

    # Run a schedule
    schedule = pipeline.schedule(
        model=ReadOnlyModel(model),
        target_node=end_node,
        requests=Requests({child_cont: ObjectSet(foo.kind(), {foo})}),
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )
    schedule.run(
        model=ReadOnlyModel(model),
        storage_provider=storage_provider,
    )

    # Check that schedule.serialize works
    schedule.serialize()

    # Check
    end_node_config = end_node.configuration_id(pipeline_configuration)
    container_location_end = ContainerLocation(2, "Container", end_node_config)
    compare_dicts(storage_provider.storage[container_location_end], {foo: b"foofoo"})

    # Run analysis
    new_model = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        analysis_name="AppendFooLibAnalysis",
        requests=Requests({child_cont: ObjectSet(foo.kind())}),
        analysis_configuration="",
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )

    # Check
    yaml_model = yaml.safe_load(bytes(new_model.serialize()))
    assert "foo.so" in yaml_model["ImportedLibraries"]


def main():
    ext, _ = import_pipebox(sys.argv[1:])
    initialize_pypeline()
    check_names(ext)
    check_pipeline()
    check_simple_pipeline()


if __name__ == "__main__":
    main()
