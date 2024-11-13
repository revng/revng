#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import sys
from tempfile import mkstemp

import requests


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


def download_file(url, local_filename):
    log(f"Downloading {local_filename}")
    try:
        with requests.get(url, stream=True) as request:
            if request.status_code == 200:
                down_fd, download_name = mkstemp(dir=os.path.dirname(local_filename))
                with open(down_fd, "wb") as debug_file:
                    for chunk in request.iter_content(chunk_size=64 * 1024):
                        debug_file.write(chunk)
                log("Downloaded")
                os.replace(download_name, local_filename)
                return True
            elif request.status_code == 404:
                log("URL was not found")
            else:
                log(f"URL returned status code {request.status_code}")
    except requests.RequestException as e:
        log(f"Exception while making request: {e}")
    return False
