#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import sys
from pathlib import Path

import requests
from xdg import xdg_cache_home


class Logger:
    def __init__(self):
        self.verbose = False

    def log(self, message):
        if self.verbose:
            sys.stderr.write(message + "\n")

    def log_warning(self, message):
        self.log("warning: " + message)


logger = Logger()
log = logger.log
log_warning = logger.log_warning


def cache_directory() -> Path:
    if "REVNG_CACHE_DIR" in os.environ:
        return Path(os.environ["REVNG_CACHE_DIR"])
    else:
        return xdg_cache_home() / "revng"


def download_file(url, local_filename):
    log(f"Downloading {local_filename}")
    with requests.get(url, stream=True) as request:
        if request.status_code == 200:
            with open(local_filename, "wb") as debug_file:
                for chunk in request.iter_content(chunk_size=64 * 1024):
                    debug_file.write(chunk)
            log("Downloaded")
            return True
        elif request.status_code == 404:
            log("URL was not found")
        else:
            log(f"URL returned status code {request.status_code}")
    return False
