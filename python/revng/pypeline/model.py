#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from typing import Annotated, Self, Set

from .object import Kind, ObjectID, ObjectSet
from .utils.cabc import ABC, abstractmethod
from .utils.registry import get_singleton

ModelPath = Annotated[str, "A string that represents a path in the model tree."]
ModelPathSet = Set[ModelPath]


class Model(ABC):
    """
    Model is an (abstract) class representing a document that configures the what a certain pipeline
    should produce.
    The document is a tree composed by dictionaries, list and scalars.
    It can be navigated via ModelPath.

    From the model, it's possible to enumerate all the objects of a certain Kind.

    The model can be changed, a change in the model might need to the invalidation of certain
    objects that have been previously produced.
    """

    @abstractmethod
    def diff(self, other: Self) -> ModelPathSet:
        raise NotImplementedError()

    @abstractmethod
    def clone(self) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def children(self, obj: ObjectID, kind: Kind) -> set[ObjectID]:
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError()

    def all_objects(self, kind: Kind) -> ObjectSet:
        """
        Returns all the objects of a certain kind in the model.
        """
        obj_id_ty: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        root: ObjectID = obj_id_ty.root()
        if kind == root.kind():
            return ObjectSet(root.kind(), {root})
        return ObjectSet(root.kind(), self.children(root, kind))

    def move_to_kind(self, objects: ObjectSet, destination_kind: Kind) -> ObjectSet:
        if not objects:
            # If the objects set is empty, we can return an empty set of the destination kind
            return ObjectSet(destination_kind)

        source_kind = objects.kind

        match destination_kind.relation(source_kind):
            case (Kind.Relation.SAME, _):
                return objects.clone()
            case (Kind.Relation.UNRELATED, _):
                return ObjectSet(destination_kind)
            case (Kind.Relation.ANCESTOR, path):
                assert isinstance(path, list)
                result = ObjectSet(destination_kind)
                for obj in objects.objects:
                    for _ in range(len(path) - 1):
                        parent = obj.parent()
                        assert parent is not None, (
                            f"Object {obj} of kind {source_kind} has no parent, "
                            f"but we are trying to move it to {destination_kind} which "
                            "is an ancestor."
                        )
                        obj = parent
                    result.add(obj)
                return result
            case (Kind.Relation.DESCENDANT, path):
                assert isinstance(path, list)

                result = objects

                # For each child kind
                for child_kind in path[:-1]:
                    # Create a new object set and populate it with the children from the previous
                    # iteration
                    children = ObjectSet(kind=child_kind)

                    for obj in result:
                        children.objects.update(self.children(obj, child_kind))

                    # Make children become the source for the next iteration
                    result = children

                assert result.kind is destination_kind
                return result

        assert False, f"Unhandled relation between kinds: {source_kind} -> {destination_kind}"

    @abstractmethod
    def serialize(self) -> bytes:
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, data: bytes) -> Model:
        pass

    @classmethod
    def is_text(cls) -> bool:
        """
        Returns True if the serialized model is a text file, False otherwise.
        This is used to determine if the model should be treated as a text file or not.
        """
        return False


class ReadOnlyModel[M: Model]:
    """
    A wrapper around the Model ensuring no changes can be made.
    """

    def __init__(self, inner: M):
        self._context: M = inner

    def diff(self, other: ReadOnlyModel[M]) -> ModelPathSet:
        return self._context.diff(other._context)  # pylint: disable=protected-access

    def clone(self) -> M:
        return self._context.clone()

    def children(self, obj: ObjectID, kind: Kind) -> ObjectSet:
        return ObjectSet(kind, self._context.children(obj, kind))

    def all_objects(self, kind: Kind) -> ObjectSet:
        return self._context.all_objects(kind)

    def move_to_kind(self, objects: ObjectSet, destination_kind: Kind) -> ObjectSet:
        return self._context.move_to_kind(objects, destination_kind)

    def serialize(self) -> bytes:
        return self._context.serialize()

    def downcast(self) -> M:
        """
        Returns the inner model, allowing to use it as a regular model.
        The user should be careful not to modify the model.
        While we could convert this to a context manager, real users will be
        written in C++ where we can enforce the immutability of the model.
        """
        return self._context
