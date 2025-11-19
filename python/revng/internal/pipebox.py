#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import signal
import tarfile
from io import BytesIO

from revng.internal.support import import_pipebox
from revng.model import Binary  # type: ignore[attr-defined]
from revng.pypeline.container import Configuration, Container
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.object import ObjectSet
from revng.pypeline.storage.file_provider import FileProvider, FileRequest
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.task.task import PipeObjectDependencies, TaskArgument, TaskArgumentAccess
from revng.support import get_root

_module, _handles = import_pipebox([get_root() / "lib/librevngPipebox.so"])


def initialize(argv: list[str] = []):
    _module.initialize(
        {
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGHUP,
            signal.SIGQUIT,
        },
        {
            signal.SIGCHLD,
            signal.SIGPIPE,
            signal.SIGUSR1,
            signal.SIGUSR2,
        },
        argv,
    )


class ImportFiles(Pipe):
    @classmethod
    def signature(cls) -> tuple[TaskArgument, ...]:
        return (
            TaskArgument(
                "binaries-container",
                _module.BinariesContainer,
                TaskArgumentAccess.WRITE,
                help_text="BinariesContainer container which will be populated",
            ),
        )

    def run(
        self,
        file_provider: FileProvider,
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeObjectDependencies:
        if len(outgoing[0]) == 0:
            return [[]]

        assert len(outgoing[0]) == 1
        root_object = list(outgoing[0].objects)[0]

        # TODO: add facilities to `model` that avoid having to serialize the
        # whole thing
        model_obj = Binary.deserialize(model.serialize().decode())

        indexes = []
        requests = []
        for binary in model_obj.Binaries:
            indexes.append(binary.Index)
            requests.append(FileRequest(binary.Hash, binary.Name, binary.Size))

        files = file_provider.get_files(requests)
        buffer = BytesIO()
        with tarfile.open(mode="w", fileobj=buffer) as tar:
            for request in requests:
                file = files[request.hash]

                info = tarfile.TarInfo()
                info.size = len(file)
                info.name = f"binaries/{request.hash}"
                info.mode = 0o644
                info.type = tarfile.REGTYPE

                tar.addfile(info, BytesIO(file))

        containers[0].deserialize({root_object: buffer.getvalue()})

        dependencies = [(root_object, f"/Binaries/{i}/Hash") for i in indexes]
        dependencies.append((root_object, "/Binaries"))

        return [dependencies]
