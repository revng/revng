#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# pylint: disable=redefined-outer-name

from __future__ import annotations

import os
from tempfile import NamedTemporaryFile
from typing import Optional, TypeVar, Union

import pytest
from simple_pipeline import ChildDictContainer, ChildObjectID, DictModel, GeneratorPipe
from simple_pipeline import InPlacePipe, NullAnalysis, PurgeAllAnalysis, PurgeOneAnalysis
from simple_pipeline import RootDictContainer, RootObjectID, SameKindPipe, ToHigherKindPipe
from simple_pipeline import ToLowerKindPipe

from revng.pypeline.container import ContainerDeclaration
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.object import ObjectID, ObjectSet
from revng.pypeline.pipeline import AnalysisBinding, Artifact, Pipeline
from revng.pypeline.pipeline_node import PipelineConfiguration, PipelineNode
from revng.pypeline.pipeline_parser import load_pipeline_yaml_file
from revng.pypeline.storage.memory import InMemoryStorageProvider
from revng.pypeline.storage.sqlite3 import SQlite3StorageProvider
from revng.pypeline.storage.storage_provider import ContainerLocation, SavePointsRange
from revng.pypeline.storage.storage_provider import StorageProvider
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.task.requests import Requests
from revng.pypeline.task.savepoint import SavePoint

Value = Union[str, int]


T = TypeVar("T")


def mandatory(arg: Optional[T]) -> T:
    assert arg is not None
    return arg


@pytest.fixture
def model():
    return DictModel()


@pytest.fixture(params=["memory", "sqlite3"])
def storage_provider(request):
    storage_provider: StorageProvider
    if request.param == "memory":
        storage_provider = InMemoryStorageProvider()
        yield storage_provider
    elif request.param == "sqlite3":
        with NamedTemporaryFile() as f:
            storage_provider = SQlite3StorageProvider(":memory:", f.name)
            yield storage_provider
    else:
        raise ValueError()


class ChildChildObjectID(ObjectID, parent_kind=ChildObjectID):
    def __init__(self, *values: str):
        self.values = values

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChildChildObjectID):
            return NotImplemented
        return self.values == other.values

    def __hash__(self) -> int:
        return hash(self.values)

    def parent(self) -> Optional[ObjectID]:
        return ChildObjectID(self.values[0]) if self.values else None

    def serialize(self) -> str:
        return "/" + self.__class__.__name__ + "/" + "/".join(self.values)

    @classmethod
    def deserialize(cls, obj: str) -> ObjectID:
        assert obj.startswith("/")
        ty, *components = obj[1:].split("/")
        assert len(components) == 2
        assert ty == cls.__name__
        return ChildChildObjectID(*components)


class Child2ObjectID(ObjectID, parent_kind=RootObjectID):
    def __init__(self, *values: str):
        self.values = values

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Child2ObjectID):
            return NotImplemented
        return self.values == other.values

    def __hash__(self) -> int:
        return hash(self.values)

    def parent(self) -> Optional[ObjectID]:
        return ChildObjectID(self.values[0]) if self.values else None

    def serialize(self) -> str:
        return "/" + self.__class__.__name__ + "/" + "/".join(self.values)

    @classmethod
    def deserialize(cls, obj: str) -> ObjectID:
        assert obj.startswith("/")
        ty, *components = obj[1:].split("/")
        assert len(components) == 2
        assert ty == cls.__name__
        return Child2ObjectID(*components)


def test_kind():
    # Test depth
    assert RootObjectID.depth() == 0
    assert ChildObjectID.depth() == 1
    assert ChildChildObjectID.depth() == 2

    # Test relation
    assert RootObjectID.relation(RootObjectID)[0] == ObjectID.Relation.SAME

    assert ChildObjectID.relation(Child2ObjectID)[0] == ObjectID.Relation.UNRELATED

    assert RootObjectID.relation(ChildObjectID) == (
        ObjectID.Relation.ANCESTOR,
        [RootObjectID, ChildObjectID],
    )
    assert ChildObjectID.relation(RootObjectID) == (
        ObjectID.Relation.DESCENDANT,
        [ChildObjectID, RootObjectID],
    )

    assert RootObjectID.relation(ChildChildObjectID) == (
        ObjectID.Relation.ANCESTOR,
        [RootObjectID, ChildObjectID, ChildChildObjectID],
    )
    assert ChildChildObjectID.relation(RootObjectID) == (
        ObjectID.Relation.DESCENDANT,
        [ChildChildObjectID, ChildObjectID, RootObjectID],
    )


def test_pipe_prerequisites_for(model) -> None:
    # These are special declarations and have to exactly match the args of the
    # pipes being tested.
    root = ObjectSet(RootObjectID, {RootObjectID()})
    one_two = ObjectSet(ChildObjectID, {ChildObjectID("one"), ChildObjectID("two")})
    one_two_three = ObjectSet(
        ChildObjectID, {ChildObjectID("one"), ChildObjectID("two"), ChildObjectID("three")}
    )
    empty_child_set = ObjectSet(ChildObjectID, set())
    empty_root_set = ObjectSet(RootObjectID, set())

    pipe: Pipe

    pipe = InPlacePipe()
    result = pipe.prerequisites_for(model, [one_two])
    assert result == [one_two]

    pipe = SameKindPipe()
    result = pipe.prerequisites_for(model, [empty_child_set, one_two])
    assert result == [one_two, empty_child_set]

    pipe = ToLowerKindPipe()
    result = pipe.prerequisites_for(model, [empty_child_set, root])
    assert result == [one_two_three, empty_root_set]

    pipe = ToHigherKindPipe()
    result = pipe.prerequisites_for(model, [empty_root_set, one_two_three])
    assert result == [root, empty_child_set]


def test_savepoint_prerequisites_for(storage_provider, model) -> None:
    storage_provider.set_model(model.serialize())
    child: ContainerDeclaration = ContainerDeclaration("child", ChildDictContainer)
    save_point = SavePoint("save", [child])

    container = ChildDictContainer()
    container.add_object(ChildObjectID("one"))
    requests = Requests({child: ObjectSet(ChildObjectID, {ChildObjectID("one")})})
    savepoint_range = SavePointsRange(start=0, end=0)
    configuration_id = "132124ujh12jk4jk124kj"

    # The storage is empty so the savepoint should be transparent
    result = save_point.prerequisites_for(
        requests=requests,
        configuration_id=configuration_id,
        storage_provider=storage_provider,
        savepoint_range=savepoint_range,
    )
    assert result == requests

    # Force the savepoint to store the object
    save_point.run(
        containers={child: container},
        incoming=requests,
        outgoing=requests,
        configuration_id=configuration_id,
        storage_provider=storage_provider,
        savepoint_range=savepoint_range,
    )

    result = save_point.prerequisites_for(
        requests=requests,
        configuration_id=configuration_id,
        storage_provider=storage_provider,
        savepoint_range=savepoint_range,
    )
    assert result == Requests({child: ObjectSet(ChildObjectID, set())})

    result = save_point.prerequisites_for(
        requests=Requests(
            {child: ObjectSet(ChildObjectID, {ChildObjectID("one"), ChildObjectID("two")})}
        ),
        configuration_id=configuration_id,
        storage_provider=storage_provider,
        savepoint_range=savepoint_range,
    )
    expected = Requests({child: ObjectSet(ChildObjectID, {ChildObjectID("two")})})
    assert result == expected


def test_pipeline_inplace(model, storage_provider):
    child_cont: ContainerDeclaration = ContainerDeclaration("arg", ChildDictContainer)
    declarations = [child_cont]
    pipeline_configuration: PipelineConfiguration = {}
    storage_provider.set_model(model.serialize())

    one = ObjectSet(ChildObjectID, {ChildObjectID("one")})
    one_two = ObjectSet(ChildObjectID, {ChildObjectID("one"), ChildObjectID("two")})

    # Create the pipeline
    begin_node = PipelineNode(SavePoint("begin", to_save=declarations))
    inplace_node = PipelineNode(InPlacePipe(), bindings=[child_cont])
    end_node = PipelineNode(SavePoint("end", to_save=declarations))
    begin_node.add_successor(inplace_node).add_successor(end_node)

    pipeline: Pipeline = Pipeline(set(declarations), begin_node)
    assert begin_node.savepoint_range == SavePointsRange(start=1, end=2)
    assert inplace_node.savepoint_range == SavePointsRange(start=1, end=2)
    assert end_node.savepoint_range == SavePointsRange(start=2, end=2)

    # Force the savepoint to store the objects
    container = ChildDictContainer()
    container.add_object(ChildObjectID("one"))
    container.add_object(ChildObjectID("two"))
    begin_configuration_id = begin_node.configuration_id(pipeline_configuration)
    begin_node.run(
        model=ReadOnlyModel(model),
        containers={child_cont: container},
        incoming=Requests({child_cont: one_two}),
        outgoing=Requests({child_cont: one_two}),
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )

    assert set(
        storage_provider.has(
            location=ContainerLocation(
                savepoint_id=1,
                container_id="arg",
                configuration_id=begin_configuration_id,
            ),
            keys=one_two,
        )
    ) == set(one_two)

    containers = pipeline.schedule(
        model=ReadOnlyModel(model),
        target_node=end_node,
        requests=Requests({child_cont: one}),
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    ).run(
        model=ReadOnlyModel(model),
        storage_provider=storage_provider,
    )
    assert containers[child_cont].objects() == one

    end_configuration_id = end_node.configuration_id(pipeline_configuration)
    assert list(
        storage_provider.has(
            location=ContainerLocation(
                savepoint_id=2,
                container_id="arg",
                configuration_id=end_configuration_id,
            ),
            keys=one,
        )
    ) == list(one)


def test_pipeline_up_down(model, storage_provider):
    root1: ContainerDeclaration = ContainerDeclaration("root_source", RootDictContainer)
    root2: ContainerDeclaration = ContainerDeclaration("root_destination", RootDictContainer)
    child1: ContainerDeclaration = ContainerDeclaration("child_destination", ChildDictContainer)
    child2: ContainerDeclaration = ContainerDeclaration("child_source", ChildDictContainer)

    declarations = [root1, root2, child1, child2]

    pipeline_configuration: PipelineConfiguration = {}
    storage_provider.set_model(model.serialize())

    begin_node = PipelineNode(SavePoint("begin", declarations))
    up_node = PipelineNode(ToHigherKindPipe(), bindings=[root1, child1])
    same_node = PipelineNode(SameKindPipe(), bindings=[child1, child2])
    down_node = PipelineNode(ToLowerKindPipe(), bindings=[child2, root2])
    end_node = PipelineNode(SavePoint("end", declarations))
    begin_node.add_successor(up_node).add_successor(same_node).add_successor(
        down_node
    ).add_successor(end_node)

    pipeline: Pipeline = Pipeline(set(declarations), begin_node)
    assert begin_node.savepoint_range == SavePointsRange(start=1, end=2)
    assert up_node.savepoint_range == SavePointsRange(start=1, end=2)
    assert same_node.savepoint_range == SavePointsRange(start=1, end=2)
    assert down_node.savepoint_range == SavePointsRange(start=1, end=2)
    assert end_node.savepoint_range == SavePointsRange(start=2, end=2)

    root_obj = ObjectSet(RootObjectID, {RootObjectID()})

    # Force the savepoint to store the objects
    container = RootDictContainer()
    container.add_object(RootObjectID())
    requests = Requests({root1: root_obj})
    begin_node.run(
        model=ReadOnlyModel(model),
        containers={root1: container},
        incoming=requests,
        outgoing=requests,
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )

    containers = pipeline.schedule(
        model=ReadOnlyModel(model),
        target_node=end_node,
        requests=Requests({root2: root_obj}),
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    ).run(
        model=ReadOnlyModel(model),
        storage_provider=storage_provider,
    )
    assert containers[root2].objects() == root_obj

    end_configuration_id = end_node.configuration_id(pipeline_configuration)
    assert set(
        storage_provider.has(
            location=ContainerLocation(
                savepoint_id=2,
                container_id="root_destination",
                configuration_id=end_configuration_id,
            ),
            keys=[RootObjectID()],
        )
    ) == {RootObjectID()}


def test_artifact(model, storage_provider):
    child1: ContainerDeclaration = ContainerDeclaration("source", ChildDictContainer)
    child2: ContainerDeclaration = ContainerDeclaration("destination", ChildDictContainer)
    declarations = [child1, child2]
    pipeline_configuration: PipelineConfiguration = {}
    one = ObjectSet.from_list([ChildObjectID("one")])
    storage_provider.set_model(model.serialize())

    begin_node = PipelineNode(SavePoint("begin", declarations))
    same_node = PipelineNode(SameKindPipe(), bindings=[child1, child2])
    begin_node.add_successor(same_node)

    artifact = Artifact("artifact", same_node, child2)

    pipeline: Pipeline = Pipeline(set(declarations), begin_node)
    assert begin_node.savepoint_range == SavePointsRange(start=1, end=1)
    assert same_node.savepoint_range == SavePointsRange(start=1, end=1)

    # Force the savepoint to store the objects
    container = ChildDictContainer()
    container.add_object(ChildObjectID("one"))
    requests = Requests({child1: one})
    begin_node.run(
        model=ReadOnlyModel(model),
        containers={child1: container},
        incoming=requests,
        outgoing=requests,
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )

    res = pipeline.get_artifact(
        model=ReadOnlyModel(model),
        artifact=artifact,
        requests=one,
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )

    assert res.objects() == one


def test_invalidation(model, storage_provider):
    # TODO: simple test: one pipe, one save point, multiple objects, invalidate one
    child: ContainerDeclaration = ContainerDeclaration("arg", ChildDictContainer)
    declarations = [child]
    pipeline_configuration: PipelineConfiguration = {}
    storage_provider.set_model(model.serialize())

    object_one = ChildObjectID("one")
    expected_output = ObjectSet(ChildObjectID, {object_one})

    pipe = PipelineNode(GeneratorPipe(), bindings=[child])
    savepoint = PipelineNode(SavePoint("end", declarations))
    pipe.add_successor(savepoint)

    pipeline: Pipeline = Pipeline(set(declarations), pipe)
    assert pipe.savepoint_range == SavePointsRange(start=0, end=1)
    assert savepoint.savepoint_range == SavePointsRange(start=1, end=1)

    containers = pipeline.schedule(
        model=ReadOnlyModel(model),
        target_node=savepoint,
        requests=Requests({child: expected_output}),
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    ).run(
        model=ReadOnlyModel(model),
        storage_provider=storage_provider,
    )
    assert containers[child].objects() == expected_output

    assert set(
        storage_provider.has(
            location=ContainerLocation(
                savepoint_id=1,
                container_id="arg",
                configuration_id=savepoint.configuration_id(pipeline_configuration),
            ),
            keys=expected_output,
        )
    ) == set(expected_output)

    storage_provider.invalidate({"/two"})

    assert set(
        storage_provider.has(
            location=ContainerLocation(
                savepoint_id=1,
                container_id="arg",
                configuration_id=savepoint.configuration_id(pipeline_configuration),
            ),
            keys=expected_output,
        )
    ) == set(expected_output)

    # Change something we depend on
    storage_provider.invalidate({"/one"})

    assert not list(
        storage_provider.has(
            location=ContainerLocation(
                savepoint_id=1,
                container_id="arg",
                configuration_id=savepoint.configuration_id(pipeline_configuration),
            ),
            keys=expected_output,
        )
    )


def test_analysis(model, storage_provider):
    # TODO: simple test: one pipe, one save point, multiple objects, invalidate one
    child: ContainerDeclaration = ContainerDeclaration("arg", ChildDictContainer)
    declarations = [child]
    pipeline_configuration: PipelineConfiguration = {}

    model["/ciao/ciao"] = "ciao"
    model["/ciao/ciao2"] = "ciao2"

    storage_provider.set_model(model.serialize())

    object_one = ChildObjectID("one")
    expected_output = ObjectSet(ChildObjectID, {object_one})

    pipe = PipelineNode(
        GeneratorPipe(),
        bindings=[child],
    )
    savepoint = PipelineNode(SavePoint("end", declarations))
    pipe.add_successor(savepoint)

    pipeline: Pipeline = Pipeline(
        set(declarations),
        pipe,
        analyses={
            AnalysisBinding(
                NullAnalysis(name="null_analysis"),
                (child,),
                savepoint,
            ),
            AnalysisBinding(
                PurgeAllAnalysis(name="purge_all_analysis"),
                (child,),
                savepoint,
            ),
            AnalysisBinding(
                PurgeOneAnalysis(name="purge_one_analysis"),
                (child,),
                savepoint,
            ),
        },
    )
    assert pipe.savepoint_range == SavePointsRange(start=0, end=1)
    assert savepoint.savepoint_range == SavePointsRange(start=1, end=1)

    orig_model = model.clone()
    new_model = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        analysis_name="null_analysis",
        requests=Requests({child: expected_output}),
        analysis_configuration="",
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )

    assert model == orig_model, "Model should not be modified by the analysis"
    assert new_model == orig_model, "This analysis doesn't invalidate anything"

    new_model = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        analysis_name="purge_all_analysis",
        requests=Requests({child: expected_output}),
        analysis_configuration="",
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )

    assert model == orig_model, "Model should not be modified by the analysis"
    assert new_model == DictModel(), "This analysis invalidates everything"

    new_model = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        analysis_name="purge_one_analysis",
        requests=Requests({child: expected_output}),
        analysis_configuration="",
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )

    assert model == orig_model, "Model should not be modified by the analysis"
    expected_model = DictModel()
    expected_model["/ciao/ciao2"] = "ciao2"
    assert new_model == expected_model, "This analysis invalidates everything"


def test_pipeline(storage_provider):
    """Load the schema and validate the pipeline.yml file against it."""
    root = os.path.dirname(os.path.abspath(__file__))
    pipeline = load_pipeline_yaml_file(os.path.join(root, "pipeline.yml"))
    pipeline_configuration: PipelineConfiguration = {}

    model = DictModel()
    with open(os.path.join(root, "model.yml"), "rb") as model_file:
        model.deserialize(model_file.read())

    res = pipeline.get_artifact(
        model=ReadOnlyModel(model),
        artifact=pipeline.artifacts["ChildArtifact"],
        requests=ObjectSet(ChildObjectID, {ChildObjectID("one")}),
        pipeline_configuration={},
        storage_provider=storage_provider,
    )
    assert res.objects() == ObjectSet(ChildObjectID, {ChildObjectID("one")})

    res = pipeline.get_artifact(
        model=ReadOnlyModel(model),
        artifact=pipeline.artifacts["RootArtifact"],
        requests=ObjectSet(RootObjectID, {RootObjectID()}),
        pipeline_configuration={},
        storage_provider=storage_provider,
    )
    assert res.objects() == ObjectSet(RootObjectID, {RootObjectID()})

    new_model = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        analysis_name="NullAnalysis",
        requests=Requests(
            {
                ContainerDeclaration(
                    name="child_destination",
                    container_type=ChildDictContainer,
                ): ObjectSet(ChildObjectID, {ChildObjectID("one")})
            }
        ),
        analysis_configuration="",
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )
    assert isinstance(new_model, DictModel), "The analysis should return the same model type"
    assert new_model == model, "NullAnalysis should not change the model"

    new_model = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        analysis_name="blackhole",  # An alias of PurgeAllAnalysis
        requests=Requests(
            {
                ContainerDeclaration(
                    name="root_source",
                    container_type=RootDictContainer,
                ): ObjectSet(RootObjectID, {RootObjectID()})
            }
        ),
        analysis_configuration="",
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )
    assert isinstance(new_model, DictModel), "The analysis should return the same model type"
    assert len(new_model) == 0, "PurgeAllAnalysis should empty the model"


def test_schedule_serdes(model):
    storage_provider = InMemoryStorageProvider()

    root1: ContainerDeclaration = ContainerDeclaration("root_source", RootDictContainer)
    root2: ContainerDeclaration = ContainerDeclaration("root_destination", RootDictContainer)
    child1: ContainerDeclaration = ContainerDeclaration("child_destination", ChildDictContainer)
    child2: ContainerDeclaration = ContainerDeclaration("child_source", ChildDictContainer)

    declarations = [root1, root2, child1, child2]

    pipeline_configuration: PipelineConfiguration = {}
    storage_provider.set_model(model.serialize())

    begin_node = PipelineNode(SavePoint("begin", declarations))
    up_node = PipelineNode(ToHigherKindPipe(), bindings=[root1, child1])
    same_node = PipelineNode(SameKindPipe(), bindings=[child1, child2])
    down_node = PipelineNode(ToLowerKindPipe(), bindings=[child2, root2])
    end_node = PipelineNode(SavePoint("end", declarations))
    begin_node.add_successor(up_node).add_successor(same_node).add_successor(
        down_node
    ).add_successor(end_node)

    pipeline: Pipeline = Pipeline(set(declarations), begin_node)
    assert begin_node.savepoint_range == SavePointsRange(start=1, end=2)
    assert up_node.savepoint_range == SavePointsRange(start=1, end=2)
    assert same_node.savepoint_range == SavePointsRange(start=1, end=2)
    assert down_node.savepoint_range == SavePointsRange(start=1, end=2)
    assert end_node.savepoint_range == SavePointsRange(start=2, end=2)

    root_obj = ObjectSet(RootObjectID, {RootObjectID()})

    # Force the savepoint to store the objects
    container = RootDictContainer()
    container.add_object(RootObjectID())
    requests = Requests({root1: root_obj})
    begin_node.run(
        model=ReadOnlyModel(model),
        containers={root1: container},
        incoming=requests,
        outgoing=requests,
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )

    schedule = pipeline.schedule(
        model=ReadOnlyModel(model),
        target_node=end_node,
        requests=Requests({root2: root_obj}),
        pipeline_configuration=pipeline_configuration,
        storage_provider=storage_provider,
    )
    schedule_str = schedule.serialize()
    pipeline.deserialize_schedule(schedule_str)
