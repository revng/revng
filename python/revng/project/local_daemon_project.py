#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import socket
from signal import SIGINT
from subprocess import DEVNULL, STDOUT, Popen
from time import sleep
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

from .daemon_project import DaemonProject
from .project import CLIProjectMixin, ResumeProjectMixin


class LocalDaemonProject(DaemonProject, CLIProjectMixin, ResumeProjectMixin):
    """
    This class extends the DaemonProject. When initialized it starts the revng daemon
    server and setups the DaemonProject client used to connect to the server.
    """

    def __init__(
        self,
        resume_path: Optional[str] = None,
        revng_executable_path: Optional[str] = None,
        *,
        connection_retries: int = 10,
    ):
        CLIProjectMixin.__init__(self, revng_executable_path)
        ResumeProjectMixin.__init__(self, resume_path)

        self._port: int = self._get_port()
        self._daemon_process: Optional[Popen] = None
        self.start_daemon(connection_retries)
        super().__init__(f"http://127.0.0.1:{self._port}/graphql/")

    def __del__(self):
        self.stop_daemon()

    def start_daemon(self, connection_retries: int):
        """
        Start the `revng` daemon and wait for it to be ready.
        """
        env = os.environ
        env["REVNG_DATA_DIR"] = self._resume_path

        cli_args = [self._revng_executable_path, "daemon", "-b", f"tcp:127.0.0.1:{self._port}"]
        self._daemon_process = Popen(cli_args, stdout=DEVNULL, stderr=STDOUT, env=env)

        failed_retries = 0
        while failed_retries <= connection_retries:
            if self._is_server_running():
                return
            failed_retries += 1
            sleep(1)
        raise RuntimeError(f"Couldn't connect to daemon server at http://127.0.0.1:{self._port}")

    def stop_daemon(self) -> int:
        """
        Stop the daemon server.
        """
        if self._daemon_process:
            self._daemon_process.send_signal(SIGINT)
            status_code = self._daemon_process.wait(30.0)
            self._daemon_process = None
            return status_code
        return 0

    def _is_server_running(self) -> bool:
        try:
            urlopen(f"http://127.0.0.1:{self._port}/status", timeout=5)
            return True
        except URLError:
            return False

    def _get_port(self) -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        free_socket = s.getsockname()[1]
        s.close()
        return int(free_socket)
