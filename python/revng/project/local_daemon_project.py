#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import socket
from pathlib import Path
from signal import SIGINT
from subprocess import DEVNULL, STDOUT, Popen
from time import sleep
from typing import Optional, Union

import requests
from requests.exceptions import RequestException
from urllib3.exceptions import HTTPError

from revng.project.common import CLIHelper
from revng.project.daemon_project import DaemonProject


class LocalDaemonProject(DaemonProject):
    """
    This class extends the DaemonProject. When initialized it starts the revng daemon
    server and setups the DaemonProject client used to connect to the server.
    """

    def __init__(
        self,
        project_directory: Union[str, Path, None] = None,
        executable_path: Optional[str] = None,
        *,
        connection_retries: int = 10,
    ):
        self._cli_helper = CLIHelper(project_directory, executable_path)
        self._host = "127.0.0.1"
        self._port: int = self._get_port()
        self._daemon_url = f"http://{self._host}:{self._port}/"
        self._daemon_process: Optional[Popen] = self._start_daemon(connection_retries)
        # We need to delay calling the parent's constructor because we first
        # needed to start the server that the parent will connect to.
        super().__init__(self._daemon_url)

    def __del__(self):
        self._stop_daemon()

    def _start_daemon(self, connection_retries: int):
        """
        Start the `revng project daemon` and wait for it to be ready.
        """
        cli_args = ["daemon", "--bind", f"{self._host}:{self._port!s}"]
        process = self._cli_helper.popen(cli_args, stdout=DEVNULL, stderr=STDOUT)

        failed_retries = 0
        while failed_retries <= connection_retries:
            if self._is_server_running():
                return process
            failed_retries += 1
            sleep(1)
        raise RuntimeError(f"Couldn't connect to daemon server at {self._daemon_url}")

    def _stop_daemon(self) -> int:
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
            req = requests.get(f"{self._daemon_url}status", timeout=5)
            return req.ok
        except (RequestException, HTTPError):
            return False

    def _get_port(self) -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        free_socket = s.getsockname()[1]
        s.close()
        return int(free_socket)
