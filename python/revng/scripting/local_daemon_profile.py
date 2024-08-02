#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import socket
import subprocess
from signal import SIGINT
from time import sleep

import requests

from revng.scripting.projects import DaemonProject


class LocalDaemonProject(DaemonProject):
    def __init__(self, resume_dir: str | None, revng_executable: str | None) -> None:
        self.port: int = self._get_port()
        self.daemon_process: subprocess.Popen | None = None
        self._set_resume_dir(resume_dir)
        self._set_revng_executable(revng_executable)
        self.start_daemon()

    def init_client(self, connection_retries: int = 10) -> None:
        failed_retries = 0
        while failed_retries <= connection_retries:
            if self.is_server_running():
                super().__init__(f"http://127.0.0.1:{self.port}")
                return
            failed_retries += 1
            sleep(1)
        raise RuntimeError(f"Couldn't connect to daemon server at http://127.0.0.1:{self.port}")

    def start_daemon(self) -> None:
        env = os.environ
        env["REVNG_DATA_DIR"] = self.resume_dir

        cli_args = [self.revng_executable, "daemon", "-b", f"tcp:127.0.0.1:{self.port}"]
        self.daemon_process = subprocess.Popen(
            cli_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env
        )

        self.init_client()

    def stop_daemon(self) -> int:
        if self.daemon_process:
            self.daemon_process.send_signal(SIGINT)
            return self.daemon_process.wait(30.0)
        raise RuntimeError("Revng daemon not running, can't stop it")

    def is_server_running(self) -> bool:
        try:
            requests.get(f"http://127.0.0.1:{self.port}/status", timeout=5)
            return True
        except requests.exceptions.ConnectionError:
            return False

    def _get_port(self) -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        free_socket = s.getsockname()[1]
        s.close()
        return int(free_socket)

    def __del__(self) -> None:
        self.stop_daemon()


class DaemonProfile:
    def __init__(self, revng_executable: str | None = None) -> None:
        self.revng_executable = revng_executable

    def get_project(self, resume_dir: str | None = None) -> LocalDaemonProject:
        return LocalDaemonProject(resume_dir, self.revng_executable)
