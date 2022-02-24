#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import logging
import secrets
from pathlib import Path

from flask import Flask, g

from revng.api import Manager
from .api import api_blueprint
from .auth import auth_blueprint, get_logged_user, login_required, is_logged_in
from .demo_webpage import demo_blueprint
from .globals import managers

REVNG_MODULE_DIR = Path(__file__).parent.parent
PIPELINES_DIR = REVNG_MODULE_DIR / "cli/translate/pipelines"

# TODO: this should be persisted somehow, otherwise:
#  - sessions will be invalidated at every server restart
#  - multiple backends are not possible (as they would have different secrets)
Flask.secret_key = secrets.token_hex(nbytes=16)

app = Flask(__name__)
app.register_blueprint(api_blueprint)
app.register_blueprint(auth_blueprint)
app.register_blueprint(demo_blueprint)


@app.before_request
def init_global_object():
    g.user = get_logged_user()
    if not g.user:
        return

    if managers.get(g.user.username) is not None:
        logging.info(f"Reusing manager for {g.user.username}")
        g.manager = managers[g.user.username]
    else:
        logging.info(f"Instantiating new manager for {g.user.username}")
        g.manager = Manager(
            flags=[],
            workdir=g.user.tmpdir,
            pipelines_paths=[
                f"{PIPELINES_DIR / 'translate.yml'}",
                f"{PIPELINES_DIR / 'isolate-translate.yml'}",
            ],
        )

    # Safety checks
    assert g.user is not None
    assert g.manager is not None
