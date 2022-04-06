#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import atexit
import logging
import secrets
from pathlib import Path
from typing import Optional

from flask import Flask, g

from revng.api._capi import initialize as capi_initialize

from .api import api_blueprint
from .demo_webpage import demo_blueprint
from .schema import SchemafulManager
from .util import project_workdir

workdir: Path = project_workdir()
manager: Optional[SchemafulManager] = None


app = Flask(__name__)
app.register_blueprint(api_blueprint)
app.register_blueprint(demo_blueprint)

app.secret_key = secrets.token_hex(16)


def cleanup():
    if manager is not None:
        store_result = manager.store_containers()
        if not store_result:
            logging.warning("Failed to store manager's containers")


atexit.register(cleanup)


@app.before_first_request
def init():
    global manager
    capi_initialize()
    manager = SchemafulManager(workdir=str(workdir.resolve()))


@app.before_request
def init_global_object():
    assert manager, "Manager not initialized"

    g.workdir = workdir
    g.manager = manager

    # Safety checks
    assert g.workdir is not None
    assert g.workdir.exists() and g.workdir.is_dir()
    assert g.manager is not None
