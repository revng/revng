#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from typing import Dict, MutableMapping, Optional, Set

from revng.pypeline.container import ContainerDeclaration, ContainerSet
from revng.pypeline.object import ObjectSet


class Requests(MutableMapping[ContainerDeclaration, ObjectSet]):
    """
    A map from a container declaration to the objects that we'd like to see there (an ObjectList).
    """

    def __init__(self, requests: Optional[Dict[ContainerDeclaration, ObjectSet]] = None):
        self.requests: Dict[ContainerDeclaration, ObjectSet] = {}
        if requests:
            self.requests.update(requests)

    def clone(self) -> Requests:
        result = Requests()
        result.requests = {k: v.clone() for k, v in self.requests.items()}
        return result

    def _requests_for(self, declaration: ContainerDeclaration) -> ObjectSet:
        """
        If the key is not present, we create a new ObjectSet for it.
        """
        kind = declaration.container_type.kind
        if declaration in self.requests:
            assert self.requests[declaration].kind == kind
        else:
            self.requests[declaration] = ObjectSet(kind=kind)

        return self.requests[declaration]

    def merge(self, other: Requests) -> None:
        for container, objects in other.items():
            self._requests_for(container).update(objects)

    def extract(self, requested_containers: Set[ContainerDeclaration]) -> Requests:
        results = Requests()

        for container, objects in list(self.requests.items()):
            if container in requested_containers:
                assert container in self.requests
                results.requests[container] = objects
                del self.requests[container]

        return results

    def check(self, containers: ContainerSet):
        """
        Check if the given containers set satisfies these requests.
        """
        # NOTE: this can be done either here or in the ContainerSet class, but
        # currently it's a dict, so it's easier to do it here
        for decl, objects in self.requests.items():
            if decl not in containers:
                raise ValueError(f"Container {decl} is not present in the given ContainerSet.")
            if not containers[decl].contains_all(objects):
                raise ValueError(
                    f"Container {containers[decl]} of declaration {decl} does "
                    f"not contain all requested objects: {objects}."
                )

    def empty(self) -> bool:
        return sum(map(len, self.requests.values())) == 0

    def __iter__(self):
        return self.requests.__iter__()

    def __getitem__(self, key: ContainerDeclaration) -> ObjectSet:
        return self.requests[key]

    def __setitem__(self, key: ContainerDeclaration, value: ObjectSet) -> None:
        if not isinstance(value, ObjectSet):
            raise TypeError(f"Expected ObjectSet, got {type(value)}")
        self.requests[key] = value

    def __delitem__(self, key: ContainerDeclaration) -> None:
        if key in self.requests:
            del self.requests[key]
        else:
            raise KeyError(f"Container {key} not found in requests.")

    def __len__(self) -> int:
        return len(self.requests)

    def insert(self, container: ContainerDeclaration, object_list: ObjectSet) -> None:
        """
        Insert a new request for the given container and object list.
        This is similar to update, but it does not check if the container already exists.
        """
        if not isinstance(object_list, ObjectSet):
            raise TypeError(f"Expected ObjectSet, got {type(object_list)}")
        self._requests_for(container).update(object_list)

    def __repr__(self):
        return repr(dict(self.requests))

    def __eq__(self, other) -> bool:
        return self.requests == other.requests

    def minimize(self) -> Requests:
        """Return the same request object, but with all the empty values removed"""
        return Requests({k: v for k, v in self.requests.items() if len(v) > 0})
