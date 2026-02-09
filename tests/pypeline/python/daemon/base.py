#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Response:
    code: int
    """The HTTP status code of the response."""
    body: Any
    """
    The response body of the request.
    """


class TestServer(ABC):
    def __init__(self):
        """Setup a temp dir where we can chwd into"""
        test_dir = Path(__file__).parent.parent
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="test_daemon")
        self.tmp_dir_path = Path(self.tmp_dir.name)
        for file in [
            "pipebox.py",
            "pipeline.yml",
            "model.yml",
        ]:
            shutil.copyfile(
                test_dir / file,
                self.tmp_dir_path / file,
            )
        self.pipebox_path = str(self.tmp_dir_path / "pipebox.py")
        self.pipeline_path = str(self.tmp_dir_path / "pipeline.yml")

    @abstractmethod
    def get_epoch(self) -> Response: ...

    @abstractmethod
    def get_pipeline(self) -> Response: ...

    @abstractmethod
    def get_model(self) -> Response: ...

    @abstractmethod
    def run_analysis(self, analysis_request) -> Response: ...

    @abstractmethod
    def get_artifact(self, artifact_request) -> Response: ...

    @abstractmethod
    def subscribe(self): ...
