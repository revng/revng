#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableSet
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional, Sequence, Set, cast, final

from revng.pypeline.graph import Graph
from revng.pypeline.utils.registry import get_registry, registry


@registry
class ObjectID(ABC):
    """
    This is the base class for identifiers of all the objects that are produced
    in the pipeline.
    We deliberately only use IDs and the concrete objects never appear in the
    library code, because we want to implement many of the Pipes in C++ so
    this decoupling allows to reduce the amount of things we need to write
    bindings for.
    To simplify talking about the relationships, we often refer to the ObjectID
    type as **ObjectKind** or more simply **Kind**.
    Each ObjectID subclass can have a parent or not, while this relationship
    is similar to the one of subclassing, it is not the same, as it has no
    implication on the substitution principle, but rather on the implicit
    dependency between the objects in the pipeline.
    """

    __slots__: tuple[str] = cast(tuple[str], ())

    __parent: type[ObjectID] | None = None
    """
    Class attributes, not instance attributes, with double underscores
    so they are not affected by subclassing.
    Subkinds are stored orthogonally to the class hierarchy, because
    the type of subkinds we are interested in has no relation with the
    substitution principle.
    """
    __children: list[type[ObjectID]] = []
    """
    Collection of all the children of this ObjectID class for convenience.
    """

    ROOT: type[ObjectID] | None = None
    """
    The root ObjectID class, i.e., the top-level class in the hierarchy.
    """

    def __init_subclass__(cls, *, parent_kind: type[ObjectID] | None = None, **kwargs):
        """
        This is where the parent relationship is established.
        When a class is subclassed, we set the parent to the class that is being
        subclassed.
        We also add the subclass to the list of children of the parent class.
        Example:
        ```python
        class RootClass(ObjectID):
            pass
        class ChildClass(ObjectID, parent_kind=RootClass):
            pass
        ```
        """
        if getattr(ObjectID, "ROOT", None) is None:
            assert parent_kind is None, "Root ObjectID class must not have a parent"
            ObjectID.ROOT = cls
        else:
            assert (
                parent_kind is not None
            ), f"There can be only one root ObjectID class, now we have {ObjectID.ROOT} and {cls}"

        cls.__parent = parent_kind
        if parent_kind is not None:
            parent_kind._register_child_kind(cls)

    @classmethod
    def parent_kind(cls) -> type[ObjectID] | None:
        """Returns the parent ObjectID class of this ObjectID class."""
        return cls.__parent

    @classmethod
    def children_kinds(cls) -> list[type[ObjectID]]:
        """Returns the list of all the children ObjectID classes of this ObjectID class."""
        return cls.__children

    @classmethod
    def _register_child_kind(cls, child_kind: type[ObjectID]) -> None:
        """
        Register a child ObjectID class to this ObjectID class.
        This is used internally to register the child classes when they are
        defined.
        """
        if child_kind not in cls.__children:
            cls.__children.append(child_kind)

    @classmethod
    def depth(cls) -> int:
        """
        Returns the depth of the ObjectID in the ObjectKind tree.
        """
        if cls.__parent is None:
            return 0
        return 1 + cls.__parent.depth()

    @classmethod
    def is_subkind_of(cls, other: type[ObjectID]) -> bool:
        """
        Returns True if this ObjectID class is a descendent of the given given
        ObjectID class, i.e., if the given ObjectID class is an ancestor in the
        ObjectKind tree.
        Also return True if the two classes are the same.
        """
        # a class is always a subkind of itself
        if cls == other:
            return True
        # if the given class is our parent, we are a subkind
        if cls.__parent == other:
            return True
        # If we have no parent, we cannot be a subkind of anything
        if cls.__parent is None:
            return False
        # If our parent is a subkind of the other, we are a subkind
        return cls.__parent.is_subkind_of(other)

    class Relation(Enum):
        SAME = 0
        ANCESTOR = 1
        DESCENDANT = 2
        UNRELATED = 3

    @classmethod
    def relation(cls, other: type[ObjectID]) -> tuple[Relation, list[type[ObjectID]] | None]:
        """
        Returns the relation between this ObjectID class and the given ObjectID class, and
        the path of classes from the descendent to the ancestor if they are related.
        """
        if cls == other:
            return (ObjectID.Relation.SAME, None)

        def find_ancestor(ancestor: type[ObjectID], start: type[ObjectID]) -> list[type[ObjectID]]:
            path: list[type[ObjectID]] = [start]
            parent = start.parent_kind()
            while parent:
                path.append(parent)
                if parent == ancestor:
                    return path
                parent = parent.parent_kind()
            return []

        # Check if cls is an ancestor of other
        path = find_ancestor(cls, other)
        if path:
            return (ObjectID.Relation.ANCESTOR, path[::-1])
        # Check if other is an ancestor of cls
        path = find_ancestor(other, cls)
        if path:
            return (ObjectID.Relation.DESCENDANT, path)

        return (ObjectID.Relation.UNRELATED, None)

    @classmethod
    def graph(
        cls,
        current_graph: Graph | None = None,
        nodes: dict[type[ObjectID], Graph.Node] | None = None,
    ) -> Graph:
        """Return the graph of the **Type Kind** hierarchy of the ObjectID
        belonging to the same tree as this class.
        The current_graph and nodes parameters are used in the recursive calls
        to populate the graph and are not generally meant to be used by the user.
        Although, this can be used to create a single graph of multiple ObjectID
        trees, as both `current_graph` and `nodes` are modified in place.
        This method can be called on any kind in a tree and it will return the
        entire tree it belongs to, including all its ancestors and descendants.
        """
        if current_graph is None:
            current_graph = Graph()
        if nodes is None:
            nodes = {}
        # if the node was already processed, we can skip it
        if cls in nodes:
            return current_graph
        # Create a new node for this class
        node = Graph.Node(cls.__name__)
        current_graph.nodes.add(node)
        nodes[cls] = node
        # Propagate to the parent
        if cls.__parent is not None:
            cls.__parent.graph(current_graph, nodes)
        # Propagate to the children
        for child in cls.__children:
            child.graph(current_graph, nodes)
            # Add the edge from this class to its children
            # Since we just processed the child, it is guaranteed to be in the
            # nodes dict
            current_graph.edges.add(Graph.Edge(node, nodes[child]))
        return current_graph

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def parent(self) -> Optional[ObjectID]:
        pass

    @abstractmethod
    def serialize(self) -> str: ...

    @classmethod
    @abstractmethod
    def deserialize(cls, obj: str) -> ObjectID:
        """Deserialize an object id of this class"""

    @staticmethod
    @final
    def deserialize_any(obj: str) -> ObjectID:
        """Deserialize an object id of any class"""
        assert obj.startswith("/")
        ty, *_components = obj[1:].split("/")
        clazz: type[ObjectID] = get_registry(ObjectID)[ty]
        return clazz.deserialize(obj)

    def __str__(self) -> str:
        return self.serialize()

    def __repr__(self) -> str:
        return repr(str(self))


# WIP: use slots
@dataclass
class ObjectSet(MutableSet[ObjectID]):
    """
    A list of objects with the same kind.
    """

    kind: type[ObjectID]
    """
    The kind of all the objects in this set.
    """

    # WIP: use SortedList
    objects: Set[ObjectID] = field(default_factory=set)
    """
    The objects in this set.
    TODO!: the user shouldn't be able to access this directly, but rather
    through the methods of this class.
    """

    @staticmethod
    def from_list(seq: Sequence[ObjectID]):
        assert len(seq) > 0
        result = ObjectSet(type(seq[0]), set(seq))
        return result

    def clone(self) -> ObjectSet:
        result = ObjectSet(self.kind)
        result.objects = set(self.objects)
        return result

    def __post_init__(self):
        for name in self.objects:
            assert isinstance(name, self.kind)

    def __contains__(self, obj: object) -> bool:
        assert isinstance(obj, self.kind)
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
        return f"ObjectSet(kind={self.kind.__name__}, objects={self.objects})"

    def add(self, value: ObjectID):
        assert isinstance(value, self.kind)
        self.objects.add(value)

    def discard(self, value: ObjectID):
        assert isinstance(value, self.kind)
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
