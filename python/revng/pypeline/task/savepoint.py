#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Sequence, final

from revng.pypeline.container import ConfigurationId, ContainerDeclaration, ContainerSet
from revng.pypeline.storage.storage_provider import ContainerLocation, SavePointsRange
from revng.pypeline.storage.storage_provider import StorageProvider

from .requests import Requests
from .task import TaskArgument, TaskArgumentAccess


class SavePoint:
    """
    A SavePoint is a task that satisfies its requests by looking up the requested objects in the
    container it owns.
    """

    __slots__: tuple = ("name", "arguments", "to_save")

    def __init__(
        self,
        name: str,
        to_save: Sequence[ContainerDeclaration],
    ):
        self.name = f"SavePoint({name or self.__class__.__name__})"
        self.arguments = [
            TaskArgument(
                name=declaration.name,
                container_type=declaration.container_type,
                access=TaskArgumentAccess.READ_WRITE,
            )
            for declaration in to_save
        ]
        self.to_save = to_save

    @final
    def prerequisites_for(
        self,
        requests: Requests,
        configuration_id: ConfigurationId,
        storage_provider: StorageProvider,
        savepoint_range: SavePointsRange,
    ) -> Requests:
        """
        This method has the same semantics as Pipe.prerequisites_for, but it has
        additional arguments that the Pipe should not have.
        """
        result = requests.clone()

        for decl, request in result.items():
            if decl not in self.to_save:
                # This savepoint does not care about this container
                continue
            # We have a match, so we can check if the objects are present
            location = ContainerLocation(
                savepoint_id=savepoint_range.start,
                container_id=decl.name,
                configuration_id=configuration_id,
            )
            found_objs = storage_provider.has(location, request)
            # If so, we should REMOVE the found objects from the request
            for found_obj in found_objs:
                result[decl].remove(found_obj)
        return result

    def run(
        self,
        containers: ContainerSet,
        incoming: Requests,
        outgoing: Requests,
        configuration_id: ConfigurationId,
        storage_provider: StorageProvider,
        savepoint_range: SavePointsRange,
    ):
        """
        The savepoint will cache the incoming containers, and then fill the outgoing
        containers with the cached data.
        """

        assert set(containers.keys()).issuperset(
            set(incoming.keys()) | set(outgoing.keys())
        ), "SavePoint containers must be a subset of incoming and outgoing requests."

        # Cache the containers present for this configuration
        for decl in self.to_save:
            if decl not in incoming:
                continue
            if len(incoming[decl]) == 0:
                continue
            storage_provider.put(
                ContainerLocation(
                    savepoint_id=savepoint_range.start,
                    container_id=decl.name,
                    configuration_id=configuration_id,
                ),
                containers[decl].serialize(incoming[decl]),
            )

        # Fill the containers from our cache
        for decl in self.to_save:
            if decl not in outgoing:
                continue
            # Empty requests do not need to be restored
            if len(outgoing[decl]) == 0:
                continue
            location = ContainerLocation(
                savepoint_id=savepoint_range.start,
                container_id=decl.name,
                configuration_id=configuration_id,
            )
            containers[decl].deserialize(storage_provider.get(location, outgoing[decl]))
