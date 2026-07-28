#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Generator, Optional, TypeVar, Union

import pytest
from pipebox import AddRootStuffAnalysis, ChildDictContainer, DictModel, GeneratorPipe
from pipebox import GeneratorPipeWithInvalidation, InPlacePipe, MyKind, MyObjectID, NullAnalysis
from pipebox import PurgeAllAnalysis, PurgeOneAnalysis, RootDictContainer, SameKindPipe
from pipebox import ToHigherKindPipe, ToLowerKindPipe

from revng.pypeline import initialize_pypeline
from revng.pypeline.container import ContainerDeclaration
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.object import Kind, ObjectSet
from revng.pypeline.pipeline import AnalysisBinding, Artifact, ArtifactCategory, Pipeline
from revng.pypeline.pipeline_node import PipelineConfiguration, PipelineNode
from revng.pypeline.pipeline_parser import load_pipeline_yaml_file
from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.schedule.scheduled_task import SavepointScheduledTask
from revng.pypeline.storage.local_provider import LocalStorageProvider
from revng.pypeline.storage.memory import InMemoryStorageProvider
from revng.pypeline.storage.storage_provider import ContainerLocation, PipeDependencies
from revng.pypeline.storage.storage_provider import SavePointsRange, StorageProvider
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.task.requests import Requests
from revng.pypeline.task.savepoint import SavePoint

# Fill the registries
initialize_pypeline()

Value = Union[str, int]


T = TypeVar("T")


def mandatory(arg: Optional[T]) -> T:
    assert arg is not None
    return arg


@pytest.fixture
def model():
    return DictModel()


@pytest.fixture(params=["memory", "local"])
def storage_provider(request, model) -> Generator[StorageProvider, None, None]:
    provider: StorageProvider
    if request.param == "memory":
        provider = InMemoryStorageProvider()
        provider.set_model(model, set(), [])
        yield provider
    elif request.param == "local":
        with NamedTemporaryFile() as f, TemporaryDirectory() as d:
            provider = LocalStorageProvider(":memory:", Path(f.name), Path(d), "")
            provider.set_model(model, set(), [])
            yield provider
    else:
        raise ValueError()


def test_kind():
    # Test rank
    assert MyKind.ROOT.rank() == 0
    assert MyKind.CHILD.rank() == 1
    assert MyKind.GRANDCHILD.rank() == 2
    assert MyKind.CHILD2.rank() == 1
    assert MyKind.root() == MyKind.ROOT
    assert MyKind.CHILD.parent() == MyKind.ROOT
    assert MyKind.GRANDCHILD.parent() == MyKind.CHILD
    assert MyKind.CHILD2.parent() == MyKind.ROOT
    assert MyKind.kinds() == [MyKind.ROOT, MyKind.CHILD, MyKind.GRANDCHILD, MyKind.CHILD2]

    # Test relation
    assert MyKind.ROOT.relation(MyKind.ROOT)[0] == Kind.Relation.SAME

    assert MyKind.CHILD.relation(MyKind.CHILD2)[0] == Kind.Relation.UNRELATED

    assert MyKind.ROOT.relation(MyKind.CHILD) == (
        Kind.Relation.ANCESTOR,
        [MyKind.ROOT, MyKind.CHILD],
    )
    assert MyKind.CHILD.relation(MyKind.ROOT) == (
        Kind.Relation.DESCENDANT,
        [MyKind.CHILD, MyKind.ROOT],
    )

    assert MyKind.ROOT.relation(MyKind.GRANDCHILD) == (
        Kind.Relation.ANCESTOR,
        [MyKind.ROOT, MyKind.CHILD, MyKind.GRANDCHILD],
    )
    assert MyKind.GRANDCHILD.relation(MyKind.ROOT) == (
        Kind.Relation.DESCENDANT,
        [MyKind.GRANDCHILD, MyKind.CHILD, MyKind.ROOT],
    )


def test_pipe_prerequisites_for(model) -> None:
    # These are special declarations and have to exactly match the args of the
    # pipes being tested.
    root = ObjectSet(MyKind.ROOT, {MyObjectID.root()})
    one_two = ObjectSet(
        MyKind.CHILD, {MyObjectID(MyKind.CHILD, "one"), MyObjectID(MyKind.CHILD, "two")}
    )
    one_two_three = ObjectSet(
        MyKind.CHILD,
        {
            MyObjectID(MyKind.CHILD, "one"),
            MyObjectID(MyKind.CHILD, "two"),
            MyObjectID(MyKind.CHILD, "three"),
        },
    )
    empty_child_set = ObjectSet(MyKind.CHILD, set())
    empty_root_set = ObjectSet(MyKind.ROOT, set())

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


def test_savepoint_prerequisites_for(storage_provider) -> None:
    child: ContainerDeclaration = ContainerDeclaration("child", ChildDictContainer)
    save_point = SavePoint("save", [child])

    container = ChildDictContainer()
    container.add_object(MyObjectID(MyKind.CHILD, "one"))
    requests = Requests({child: ObjectSet(MyKind.CHILD, {MyObjectID(MyKind.CHILD, "one")})})
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
        pipes_dependencies=[],
    )

    result = save_point.prerequisites_for(
        requests=requests,
        configuration_id=configuration_id,
        storage_provider=storage_provider,
        savepoint_range=savepoint_range,
    )
    assert result == Requests({child: ObjectSet(MyKind.CHILD, set())})

    result = save_point.prerequisites_for(
        requests=Requests(
            {
                child: ObjectSet(
                    MyKind.CHILD, {MyObjectID(MyKind.CHILD, "one"), MyObjectID(MyKind.CHILD, "two")}
                )
            }
        ),
        configuration_id=configuration_id,
        storage_provider=storage_provider,
        savepoint_range=savepoint_range,
    )
    expected = Requests({child: ObjectSet(MyKind.CHILD, {MyObjectID(MyKind.CHILD, "two")})})
    assert result == expected


def test_pipeline_inplace(model, storage_provider):
    child_cont: ContainerDeclaration = ContainerDeclaration("arg", ChildDictContainer)
    declarations = [child_cont]
    configuration: PipelineConfiguration = {}

    one = ObjectSet(MyKind.CHILD, {MyObjectID(MyKind.CHILD, "one")})
    one_two = ObjectSet(
        MyKind.CHILD, {MyObjectID(MyKind.CHILD, "one"), MyObjectID(MyKind.CHILD, "two")}
    )

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
    container.add_object(MyObjectID(MyKind.CHILD, "one"))
    container.add_object(MyObjectID(MyKind.CHILD, "two"))
    begin_configuration_id = begin_node.configuration_id(configuration)
    task = SavepointScheduledTask(
        begin_node,
        ReadOnlyModel(model),
        storage_provider,
        configuration,
        (Requests({child_cont: one_two}), Requests({child_cont: one_two})),
    )
    task.run({child_cont: container}, RunnerContext())

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
        configuration=configuration,
        storage_provider=storage_provider,
    ).run()
    assert containers[child_cont].objects() == one

    end_configuration_id = end_node.configuration_id(configuration)
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

    configuration: PipelineConfiguration = {}

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

    root_obj = ObjectSet(MyKind.ROOT, {MyObjectID.root()})

    # Force the savepoint to store the objects
    container = RootDictContainer()
    container.add_object(MyObjectID.root())
    requests = Requests({root1: root_obj})
    task = SavepointScheduledTask(
        begin_node,
        ReadOnlyModel(model),
        storage_provider,
        configuration,
        (requests, requests),
    )
    task.run({root1: container}, RunnerContext())

    containers = pipeline.schedule(
        model=ReadOnlyModel(model),
        target_node=end_node,
        requests=Requests({root2: root_obj}),
        configuration=configuration,
        storage_provider=storage_provider,
    ).run()
    assert containers[root2].objects() == root_obj

    end_configuration_id = end_node.configuration_id(configuration)
    assert set(
        storage_provider.has(
            location=ContainerLocation(
                savepoint_id=2,
                container_id="root_destination",
                configuration_id=end_configuration_id,
            ),
            keys=[MyObjectID.root()],
        )
    ) == {MyObjectID.root()}


def test_artifact(model, storage_provider):
    child1: ContainerDeclaration = ContainerDeclaration("source", ChildDictContainer)
    child2: ContainerDeclaration = ContainerDeclaration("destination", ChildDictContainer)
    declarations = [child1, child2]
    configuration: PipelineConfiguration = {}
    one = ObjectSet.from_list([MyObjectID(MyKind.CHILD, "one")])

    begin_node = PipelineNode(SavePoint("begin", declarations))
    same_node = PipelineNode(SameKindPipe(), bindings=[child1, child2])
    begin_node.add_successor(same_node)

    artifact = Artifact("artifact", same_node, child2, ArtifactCategory("main", True))

    pipeline: Pipeline = Pipeline(set(declarations), begin_node)
    assert begin_node.savepoint_range == SavePointsRange(start=1, end=1)
    assert same_node.savepoint_range == SavePointsRange(start=1, end=1)

    # Force the savepoint to store the objects
    container = ChildDictContainer()
    container.add_object(MyObjectID(MyKind.CHILD, "one"))
    requests = Requests({child1: one})
    task = SavepointScheduledTask(
        begin_node,
        ReadOnlyModel(model),
        storage_provider,
        configuration,
        (requests, requests),
    )
    task.run({child1: container}, RunnerContext())

    res = pipeline.get_artifact(
        model=ReadOnlyModel(model),
        artifact=artifact,
        requests=one,
        configuration=configuration,
        storage_provider=storage_provider,
    )

    assert res.objects() == one


def test_invalidation(model, storage_provider):
    # TODO: simple test: one pipe, one save point, multiple objects, invalidate one
    child: ContainerDeclaration = ContainerDeclaration("arg", ChildDictContainer)
    declarations = [child]
    configuration: PipelineConfiguration = {}

    object_one = MyObjectID(MyKind.CHILD, "one")
    expected_output = ObjectSet(MyKind.CHILD, {object_one})

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
        configuration=configuration,
        storage_provider=storage_provider,
    ).run()
    assert containers[child].objects() == expected_output

    assert set(
        storage_provider.has(
            location=ContainerLocation(
                savepoint_id=1,
                container_id="arg",
                configuration_id=savepoint.configuration_id(configuration),
            ),
            keys=expected_output,
        )
    ) == set(expected_output)

    storage_provider.set_model(storage_provider.get_model()[0], {"/two"}, [])

    assert set(
        storage_provider.has(
            location=ContainerLocation(
                savepoint_id=1,
                container_id="arg",
                configuration_id=savepoint.configuration_id(configuration),
            ),
            keys=expected_output,
        )
    ) == set(expected_output)

    # Change something we depend on
    storage_provider.set_model(storage_provider.get_model()[0], {"/one"}, [])

    assert not list(
        storage_provider.has(
            location=ContainerLocation(
                savepoint_id=1,
                container_id="arg",
                configuration_id=savepoint.configuration_id(configuration),
            ),
            keys=expected_output,
        )
    )


def test_analysis(model, storage_provider):
    # TODO: simple test: one pipe, one save point, multiple objects, invalidate one
    child: ContainerDeclaration = ContainerDeclaration("arg", ChildDictContainer)
    declarations = [child]
    configuration: PipelineConfiguration = {}

    model["/test/test"] = "test"
    model["/test/test2"] = "test2"
    storage_provider.set_model(model, set(), [])

    object_one = MyObjectID(MyKind.CHILD, "one")
    expected_output = ObjectSet(MyKind.CHILD, {object_one})

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
                NullAnalysis(),
                (child,),
                savepoint,
            ),
            AnalysisBinding(
                PurgeAllAnalysis(),
                (child,),
                savepoint,
            ),
            AnalysisBinding(
                PurgeOneAnalysis(),
                (child,),
                savepoint,
            ),
        },
    )
    assert pipe.savepoint_range == SavePointsRange(start=0, end=1)
    assert savepoint.savepoint_range == SavePointsRange(start=1, end=1)

    orig_model = model.clone()
    new_model, invalidated = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        analysis_name="NullAnalysis",
        requests=Requests({child: expected_output}),
        configuration=configuration,
        storage_provider=storage_provider,
    )

    assert model == orig_model, "Model should not be modified by the analysis"
    assert new_model == orig_model, "This analysis doesn't invalidate anything"
    assert not invalidated, "This analysis doesn't invalidate anything"

    new_model, invalidated = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        analysis_name="PurgeAllAnalysis",
        requests=Requests({child: expected_output}),
        configuration=configuration,
        storage_provider=storage_provider,
    )

    assert model == orig_model, "Model should not be modified by the analysis"
    assert new_model == DictModel(), "This analysis invalidates everything"

    new_model, invalidated = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        analysis_name="PurgeOneAnalysis",
        requests=Requests({child: expected_output}),
        configuration=configuration,
        storage_provider=storage_provider,
    )

    assert model == orig_model, "Model should not be modified by the analysis"
    expected_model = DictModel()
    expected_model["/test/test2"] = "test2"
    assert new_model == expected_model, "This analysis invalidates everything"


def test_pipeline(storage_provider):
    """Load the schema and validate the pipeline.yml file against it."""
    root = os.path.dirname(os.path.abspath(__file__))
    pipeline = load_pipeline_yaml_file(os.path.join(root, "pipeline.yml"))
    configuration: PipelineConfiguration = {}

    with open(os.path.join(root, "model.yml"), "rb") as model_file:
        model = DictModel.deserialize(model_file.read())[0]

    res = pipeline.get_artifact(
        model=ReadOnlyModel(model),
        artifact=pipeline.artifacts["ChildArtifact"],
        requests=ObjectSet(MyKind.CHILD, {MyObjectID(MyKind.CHILD, "one")}),
        configuration={},
        storage_provider=storage_provider,
    )
    assert res.objects() == ObjectSet(MyKind.CHILD, {MyObjectID(MyKind.CHILD, "one")})

    res = pipeline.get_artifact(
        model=ReadOnlyModel(model),
        artifact=pipeline.artifacts["RootArtifact"],
        requests=ObjectSet(MyKind.ROOT, {MyObjectID.root()}),
        configuration={},
        storage_provider=storage_provider,
    )
    assert res.objects() == ObjectSet(MyKind.ROOT, {MyObjectID.root()})

    new_model, invalidated = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        analysis_name="NullAnalysis",
        requests=Requests(
            {
                ContainerDeclaration(
                    name="child_destination",
                    container_type=ChildDictContainer,
                ): ObjectSet(MyKind.CHILD, {MyObjectID(MyKind.CHILD, "one")})
            }
        ),
        configuration=configuration,
        storage_provider=storage_provider,
    )
    assert isinstance(new_model, DictModel), "The analysis should return the same model type"
    assert new_model == model, "NullAnalysis should not change the model"

    new_model, invalidated = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        # An alias of PurgeAllAnalysis
        analysis_name="PurgeAllAnalysis",
        requests=Requests(
            {
                ContainerDeclaration(
                    name="root_source",
                    container_type=RootDictContainer,
                ): ObjectSet(MyKind.ROOT, {MyObjectID.root()})
            }
        ),
        configuration=configuration,
        storage_provider=storage_provider,
    )
    assert isinstance(new_model, DictModel), "The analysis should return the same model type"
    assert len(new_model) == 0, "PurgeAllAnalysis should empty the model"
    stored_model, epoch = storage_provider.get_model()
    assert stored_model == new_model, "The model should be stored"


def test_schedule_serdes(model):
    storage_provider = InMemoryStorageProvider()

    root1: ContainerDeclaration = ContainerDeclaration("root_source", RootDictContainer)
    root2: ContainerDeclaration = ContainerDeclaration("root_destination", RootDictContainer)
    child1: ContainerDeclaration = ContainerDeclaration("child_destination", ChildDictContainer)
    child2: ContainerDeclaration = ContainerDeclaration("child_source", ChildDictContainer)

    declarations = [root1, root2, child1, child2]

    configuration: PipelineConfiguration = {}

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

    root_obj = ObjectSet(MyKind.ROOT, {MyObjectID.root()})

    # Force the savepoint to store the objects
    container = RootDictContainer()
    container.add_object(MyObjectID.root())
    requests = Requests({root1: root_obj})
    task = SavepointScheduledTask(
        begin_node,
        ReadOnlyModel(model),
        storage_provider,
        configuration,
        (requests, requests),
    )
    task.run({root1: container}, RunnerContext())

    schedule = pipeline.schedule(
        model=ReadOnlyModel(model),
        target_node=end_node,
        requests=Requests({root2: root_obj}),
        configuration=configuration,
        storage_provider=storage_provider,
    )
    schedule_str = schedule.serialize()
    pipeline.deserialize_schedule(schedule_str, ReadOnlyModel(model), storage_provider)


def test_storage_invalidation(storage_provider: StorageProvider):
    configuration_id = ""
    container_id = "a"

    """
    Structure of the savepoints IDs used in this test

        0
    v---|---v
    1       2
            |
            v
            4

    1, 4 - root
    2, 3 - function
    """

    root = MyObjectID(MyKind.ROOT)
    function1 = MyObjectID(MyKind.CHILD, "func1")
    function2 = MyObjectID(MyKind.CHILD, "func2")

    def add_object(save_start: int, save_end: int, object_id, path: str):
        storage_provider.add_objects(
            [
                PipeDependencies(
                    0,
                    SavePointsRange(save_start, save_end),
                    configuration_id,
                    [(container_id, object_id, path)],
                    [],
                )
            ],
            {ContainerLocation(save_start, container_id, configuration_id): {object_id: b""}},
        )

    # Check basic invalidation
    add_object(0, 4, root, "/root")
    result = storage_provider.set_model(storage_provider.get_model()[0], {"/root"}, [])
    assert result.invalidated_objects == {
        ContainerLocation(0, container_id, configuration_id): {root}
    }

    # Check downward invalidation
    storage_provider.prune_objects()
    add_object(0, 4, root, "/root")
    add_object(2, 4, function1, "/function1")
    result = storage_provider.set_model(storage_provider.get_model()[0], {"/root"}, [])
    assert result.invalidated_objects == {
        ContainerLocation(0, container_id, configuration_id): {root},
        ContainerLocation(2, container_id, configuration_id): {function1},
    }

    # Check upward invalidation
    storage_provider.prune_objects()
    add_object(0, 4, root, "/root")
    add_object(2, 4, function1, "/function1")
    add_object(2, 4, function2, "/function2")
    add_object(3, 3, root, "/root")
    result = storage_provider.set_model(storage_provider.get_model()[0], {"/function1"}, [])
    assert result.invalidated_objects == {
        ContainerLocation(2, container_id, configuration_id): {function1},
        ContainerLocation(3, container_id, configuration_id): {root},
    }


def test_disk_model_change_targeted_invalidation():
    container_id = "a"
    configuration_id = ""
    obj_one = MyObjectID(MyKind.CHILD, "one")
    obj_two = MyObjectID(MyKind.CHILD, "two")

    with NamedTemporaryFile() as f, TemporaryDirectory() as d:
        model_path = Path(f.name)

        # Write the baseline model on disk and record it in the provider
        baseline = DictModel()
        baseline["/one"] = 1
        baseline["/two"] = 2
        model_path.write_bytes(baseline.serialize())

        provider = LocalStorageProvider(":memory:", model_path, Path(d), "")
        provider.set_model(baseline, set(), [])

        # Nothing changed on disk yet
        assert not provider.unprocessed_model_changes()[0]

        def add_object(object_id, path):
            provider.add_objects(
                [
                    PipeDependencies(
                        0,
                        SavePointsRange(0, 0),
                        configuration_id,
                        [(container_id, object_id, path)],
                        [],
                    )
                ],
                {ContainerLocation(0, container_id, configuration_id): {object_id: b""}},
            )

        add_object(obj_one, "/one")
        add_object(obj_two, "/two")

        # Edit the model on disk, changing only "/one"
        changed = DictModel()
        changed["/one"] = 42
        changed["/two"] = 2
        model_path.write_bytes(changed.serialize())
        # Force a distinct mtime so the cheap gate detects the change
        stat = model_path.stat()
        os.utime(model_path, (stat.st_atime, stat.st_mtime + 100))

        # The diff points exactly at the changed path
        diff, _ = provider.unprocessed_model_changes()
        assert diff
        assert diff.paths() == {"/one"}

        # Applying the change purges only the object that depended on "/one"
        result = provider.set_model(changed, diff.paths(), [])
        assert result.invalidated_objects == {
            ContainerLocation(0, container_id, configuration_id): {obj_one}
        }
        assert set(
            provider.has(ContainerLocation(0, container_id, configuration_id), [obj_two])
        ) == {obj_two}

        # The baseline is now up to date, no further changes are reported
        assert not provider.unprocessed_model_changes()[0]


def test_external_model_change_preserves_on_disk_bytes():
    container_id = "a"
    configuration_id = ""
    obj_one = MyObjectID(MyKind.CHILD, "one")
    obj_two = MyObjectID(MyKind.CHILD, "two")

    with NamedTemporaryFile() as f, TemporaryDirectory() as d:
        model_path = Path(f.name)

        baseline = DictModel()
        baseline["/one"] = 1
        baseline["/two"] = 2
        model_path.write_bytes(baseline.serialize())

        provider = LocalStorageProvider(":memory:", model_path, Path(d), "")
        provider.set_model(baseline, set(), [])
        # Record the baseline before editing so the change is detected below.
        assert not provider.unprocessed_model_changes()[0]

        def add_object(object_id, path):
            provider.add_objects(
                [
                    PipeDependencies(
                        0,
                        SavePointsRange(0, 0),
                        configuration_id,
                        [(container_id, object_id, path)],
                        [],
                    )
                ],
                {ContainerLocation(0, container_id, configuration_id): {object_id: b""}},
            )

        add_object(obj_one, "/one")
        add_object(obj_two, "/two")

        # Hand-edit the model on disk in a non-canonical form (a comment and a
        # different key order) that still parses to a change of only "/one".
        changed = DictModel()
        changed["/one"] = 42
        changed["/two"] = 2
        non_canonical = b"# hand-written\n/two: 2\n/one: 42\n"
        assert non_canonical != changed.serialize()
        model_path.write_bytes(non_canonical)
        stat = model_path.stat()
        os.utime(model_path, (stat.st_atime, stat.st_mtime + 100))

        diff, model_bytes = provider.unprocessed_model_changes()
        assert diff.paths() == {"/one"}

        # Reconciling an external change purges only the object that depended on
        # "/one", exactly like a normal set_model...
        result = provider.set_model(changed, diff.paths(), [], model_bytes)
        assert result.invalidated_objects == {
            ContainerLocation(0, container_id, configuration_id): {obj_one}
        }
        assert set(
            provider.has(ContainerLocation(0, container_id, configuration_id), [obj_two])
        ) == {obj_two}

        # ...but the on-disk file is kept verbatim, not canonicalized.
        assert model_path.read_bytes() == non_canonical
        # And the baseline is up to date, so no further changes are reported.
        assert not provider.unprocessed_model_changes()[0]


def test_custom_invalidation(model, storage_provider: StorageProvider):
    # Create a simple pipeline with one pipe, one savepoint and an analysis
    # attached to the savepoint
    root_decl: ContainerDeclaration = ContainerDeclaration("root", RootDictContainer)
    declarations = [root_decl]

    configuration: PipelineConfiguration = {}

    begin_node = PipelineNode(GeneratorPipeWithInvalidation(), bindings=[root_decl])
    savepoint_node = PipelineNode(SavePoint("end", declarations))
    begin_node.add_successor(savepoint_node)

    analyses = {AnalysisBinding(AddRootStuffAnalysis(), (root_decl,), savepoint_node)}

    pipeline: Pipeline = Pipeline(set(declarations), begin_node, None, analyses)
    root_obj = ObjectSet(MyKind.ROOT, {MyObjectID.root()})

    # Trigger the creation of the root object in the savepoint, this will run the
    # `GeneratorPipeWithInvalidation` to produce it and will store custom
    # invalidation in storage
    schedule = pipeline.schedule(
        model=ReadOnlyModel(model),
        target_node=savepoint_node,
        requests=Requests({root_decl: root_obj}),
        configuration=configuration,
        storage_provider=storage_provider,
    )
    schedule.run()

    # Run an analysis that changes the model (so a real model diff is persisted)
    # and retrieve the invalidated objects. `GeneratorPipeWithInvalidation`
    # reads no model fields, so `ROOT` can only be invalidated through custom
    # invalidation, which is exactly what we want to observe.
    _, invalidated = pipeline.run_analysis(
        model=ReadOnlyModel(model),
        analysis_name=AddRootStuffAnalysis.name,
        requests=Requests({root_decl: root_obj}),
        configuration=configuration,
        storage_provider=storage_provider,
    )

    # Check that the invalidation is what we expect. There should be `ROOT`
    # invalidated in the only savepoint present.
    begin_configuration_id = begin_node.configuration_id(configuration)
    location = ContainerLocation(1, root_decl.name, begin_configuration_id)
    assert {location} == set(invalidated.keys())
    assert invalidated[location] == {MyObjectID.root()}
