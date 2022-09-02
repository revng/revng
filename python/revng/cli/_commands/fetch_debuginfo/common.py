#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import gzip
import os
import shutil
import sys

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
    with requests.get(url, stream=True) as r:
        if r.status_code == 200:
            # TODO: Add support for other types of compressed files.
            if "Content-Encoding" in r.headers and r.headers["Content-Encoding"] == "gzip":
                gzip_filename = local_filename + ".gz"
                with gzip.open(gzip_filename, "wb") as gzip_file:
                    shutil.copyfileobj(r.raw, gzip_file)
                with gzip.open(gzip_filename, "rb") as gzip_file:
                    content = gzip.decompress(gzip_file.read())
                with open(local_filename, "wb") as debug_file:
                    debug_file.write(content)
                os.remove(gzip_filename)
            else:
                with open(local_filename, "wb") as debug_file:
                    shutil.copyfileobj(r.raw, debug_file)

            log("Downloaded")
            return True
        else:
            log("URL was not found")
            return False
    return False
