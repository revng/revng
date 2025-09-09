#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from collections.abc import MutableSet
from dataclasses import dataclass, field
from enum import Enum

# builtin since python 3.9
from graphlib import TopologicalSorter
from typing import Iterator, Optional, Sequence, Set, cast

from .graph import Graph
from .utils.cabc import ABC, abstractmethod


class Kind(ABC):
    __name__: str
    # Class attributes set in __init_subclass__
    _ranks: dict[Kind, int]
    _children: dict[Kind, set[Kind]]
    _root: Kind | None

    # These two methods define the hierarchy
    @classmethod
    @abstractmethod
    def kinds(cls) -> list[Kind]:
        """Return a list of all the kinds"""
        raise NotImplementedError()

    @abstractmethod
    def parent(self) -> Kind | None:
        """Get the parent of this kind, or None if root"""
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        """Standard python hash, needed to put Kinds in sets and dicts"""
        raise NotImplementedError()

    # Back and forth from str to kind for storage provider and CLI
    @abstractmethod
    def serialize(self) -> str:
        """Convert the kind to a string representation that can be later be deserialized"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def deserialize(cls, value: str) -> Kind:
        """Convert a string representation back to a kind"""
        raise NotImplementedError()

    # Methods we can provide concrete implementations based on the above methods
    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __str__(self) -> str:
        return self.serialize()

    def __repr__(self) -> str:
        return self.serialize()

    @classmethod
    def _init_type(cls):
        """
        Verify that `all` and `parent` are at reasonable and compute useful
        things like ranks, children, and the root.
        """
        kinds = set(cls.kinds())
        assert len(kinds) > 0, "There must be at least one kind"
        cls._ranks: dict[Kind, int] = {}
        """Cache of the rank of each kind"""
        cls._children: dict[Kind, set[Kind]] = {}
        """Cache of the children of each kind"""
        cls._root: Kind | None = None
        """The root of the kinds"""
        for kind in kinds:
            # Check that it's reasonable
            assert isinstance(kind, Kind)
            parent = kind.parent()
            if parent is None:
                assert cls._root is None, f"Found two roots! {cls._root} and {kind}"
                cls._root = kind
                cls._ranks[cls._root] = 0
            else:
                cls._children.setdefault(parent, set()).add(kind)
                assert parent in kinds, (
                    "The parent of each kind should be a known kind. "
                    f"Got parent {parent} which is not in the "
                    f"known kinds '{kinds}'"
                )
        assert (
            cls._root is not None
        ), "Could not find a root, there must be a loop in the hierarchy."
        # Assign ranks
        ts = TopologicalSorter(cls._children)
        for kind in reversed(list(ts.static_order())):
            parent = kind.parent()
            if parent is None:
                continue
            cls._ranks[kind] = cls._ranks[parent] + 1

    def children(self) -> set[Kind]:
        """
        Return all the children of this kind, this is a generic impl
        and the implementer can probably write a more efficient one.
        """
        return self.__class__._children.get(self, set())

    def rank(self) -> int:
        """
        Return the distance of the current kind from the root.
        """
        return self.__class__._ranks[self]

    @classmethod
    def root(cls) -> Kind:
        # We already check in the __init_subclass__ that there is a root
        return cast(Kind, cls._root)

    def is_subkind_of(self, other: Kind) -> bool:
        """
        Returns whether the current kind is equal
        to `other` or it's a descendent of `other`
        """
        if self == other:
            return True
        parent = self.parent()
        if parent is None:
            return False
        return parent.is_subkind_of(other)

    @classmethod
    def graph(
        cls,
    ) -> Graph:
        """
        Returns a graph of the full kinds hierarchy for printing porpouses
        """
        graph = Graph()
        nodes = {}
        # Create all nodes
        for kind in cls.kinds():
            node = Graph.Node(kind.__name__)
            graph.nodes.add(node)
            nodes[kind] = node
        # Create all edges
        for kind in cls.kinds():
            parent = kind.parent()
            # The root can't have arcs
            if parent is None:
                continue
            graph.edges.add(Graph.Edge(nodes[parent], nodes[kind]))
        return graph

    class Relation(Enum):
        SAME = 0
        ANCESTOR = 1
        DESCENDANT = 2
        UNRELATED = 3

    def relation(self, other: Kind) -> tuple[Relation, list[Kind] | None]:
        """
        Returns the relation between the two kinds, and, if related, returns
        the path to get from the ancestor to the descendent.
        """
        # Easy case
        if self == other:
            return Kind.Relation.SAME, None
        # Same rank but different means always unrelated
        if self.rank() == other.rank():
            return Kind.Relation.UNRELATED, None
        # Start from descendant and raise up to ancestor
        ancestor = min(self, other, key=lambda x: x.rank())
        descendant = max(self, other, key=lambda x: x.rank())
        path = [descendant]
        while descendant != ancestor:
            parent = descendant.parent()
            # If we got to the root, the other node must be on a different
            # branch, thus unrelated
            if parent is None:
                return Kind.Relation.UNRELATED, None
            path.append(parent)
            descendant = parent

        if self == ancestor:
            return Kind.Relation.ANCESTOR, path[::-1]
        else:
            return Kind.Relation.DESCENDANT, path


class ObjectID(ABC):
    # Needed for model's `move_to_kind`
    @abstractmethod
    def kind(self) -> Kind:
        """Return the kind of this object"""
        raise NotImplementedError()

    @abstractmethod
    def parent(self) -> Optional[ObjectID]:
        """Return the parent object of this object, or None if root"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def root(cls) -> ObjectID:
        """Return an instance of the root object of this hierarchy"""
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        """Standard python hash, needed to put ObjectIDs in sets and dicts"""
        raise NotImplementedError()

    @abstractmethod
    def serialize(self) -> str:
        """Convert the object to a string representation that can be later be deserialized"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def deserialize(cls, obj: str) -> ObjectID:
        """Convert a string representation back to an object"""
        raise NotImplementedError()

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __str__(self) -> str:
        return self.serialize()

    def __repr__(self) -> str:
        return repr(str(self))


@dataclass(slots=True)
class ObjectSet(MutableSet[ObjectID]):
    """
    A list of objects with the same kind.
    """

    kind: Kind
    """
    The kind of all the objects in this set.
    """

    objects: Set[ObjectID] = field(default_factory=set)
    """
    The objects in this set.
    TODO!: the user shouldn't be able to access this directly, but rather
    through the methods of this class.
    """

    @staticmethod
    def from_list(seq: Sequence[ObjectID]):
        assert len(seq) > 0
        result = ObjectSet(seq[0].kind(), set(seq))
        return result

    def clone(self) -> ObjectSet:
        result = ObjectSet(self.kind)
        result.objects = set(self.objects)
        return result

    def __post_init__(self):
        for obj in self.objects:
            assert isinstance(obj, ObjectID), f"Expected ObjectID, got {obj}"
            assert obj.kind() == self.kind

    def __contains__(self, obj: object) -> bool:
        assert isinstance(obj, ObjectID)
        assert obj.kind() == self.kind
        return obj in self.objects

    def __iter__(self) -> Iterator[ObjectID]:
        return self.objects.__iter__()

    def __len__(self) -> int:
        return len(self.objects)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectSet):
            return False
        if self.kind != other.kind:
            return False
        return self.objects == other.objects

    def __repr__(self) -> str:
        return f"ObjectSet(kind={self.kind.serialize()}, objects={self.objects})"

    def add(self, value: ObjectID):
        assert value.kind() == self.kind
        self.objects.add(value)

    def discard(self, value: ObjectID):
        assert value.kind() == self.kind
        self.objects.discard(value)

    def update(self, *others: ObjectSet):
        for other in others:
            if not isinstance(other, ObjectSet):
                raise TypeError(f"Expected ObjectSet, got {type(other)}")
            if other.kind != self.kind:
                raise ValueError(
                    f"Cannot update ObjectSet of kind {self.kind} with ObjectSet"
                    f" of kind {other.kind}."
                )
            self.objects.update(other.objects)

    def issubset(self, other: ObjectSet) -> bool:
        return self <= other
