#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import logging
from typing import TYPE_CHECKING
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from revng.api.exceptions import RevngException
from revng.api.manager import Manager
from revng.api.rank import Rank
from .auth import login_required
from .util import clean_dict, clean_double_dict
from .validation_utils import all_strings

if TYPE_CHECKING:
    from flask.ctx import _AppCtxGlobals

    class FlaskGlobals(_AppCtxGlobals):
        manager: Manager

    g = FlaskGlobals()
else:
    from flask import g


api_blueprint = Blueprint("api", __name__)


def json_response(data=None):
    if data is None:
        data = {}
    return jsonify(data)


def json_error(message, http_code=404):
    data = {
        "error": message,
    }
    return jsonify(data), http_code


@api_blueprint.route("/data/<step_name>/<container_name>/<path:pathspec>")
@login_required
def produce(step_name: str, container_name: str, pathspec: str):
    # kind_name = request.json["kind"]
    # exact = bool(request.json.get("exact", True))
    # path_components = request.json["path_components"]
    try:
        product = g.manager.produce_target(pathspec, step_name, container_name)
        return json_response(product)
    except RevngException as e:
        return json_error(repr(e))


@api_blueprint.post("/bulk-data")
@login_required
def bulk_produce():
    step_name = request.json["step"]
    container_name = request.json["container"]
    targets = request.json["targets"]
    if not all_strings(targets):
        return json_error("Targets must be all strings")
    try:
        product = g.manager.produce_target(targets, step_name, container_name)
        return json_response(product)
    except RevngException as e:
        return json_error(repr(e))


@api_blueprint.route("/step/<step_name>/<container_name>/targets")
@login_required
def step_container_targets(step_name, container_name):
    """Returns the list of targets for the given container/step tuple"""
    try:
        targets = g.manager.get_targets(step_name, container_name)
        return json_response([t.as_dict() for t in targets])
    except RevngException as e:
        return json_error(repr(e))


@api_blueprint.route("/step/<step_name>/targets")
@login_required
def step_targets(step_name):
    """Returns the list of targets for the given step"""
    try:
        grouped_targets = g.manager.get_targets_from_step(step_name)
        response = {k: [t.as_dict() for t in v] for k, v in grouped_targets.items()}
        clean_dict(response)
        return json_response(response)
    except RevngException as e:
        return json_error(repr(e))


@api_blueprint.route("/step/targets")
@login_required
def all_targets():
    try:
        targets = g.manager.get_all_targets()
        response = {
            k: {k2: [t.as_dict() for t in v2] for k2, v2 in v.items()} for k, v in targets.items()
        }
        clean_double_dict(response)
        return json_response(response)
    except RevngException as e:
        return json_error(repr(e))


@api_blueprint.route("/find-targets/", strict_slashes=False)
@login_required
def find_targets_blank():
    return find_targets("")


@api_blueprint.route("/find-targets/<path:pathspec>")
@login_required
def find_targets(pathspec: str):
    logging.warn(pathspec)
    try:
        targets = g.manager.get_all_targets()
        result = {
            k: {
                k2: [t.as_dict() for t in v2 if t.joined_path() == pathspec] for k2, v2 in v.items()
            }
            for k, v in targets.items()
        }
        clean_double_dict(result)
        return json_response(result)
    except RevngException as e:
        return json_error(repr(e))


@api_blueprint.route("/data/<step_name>")
@login_required
def get_step_containers(step_name: str):
    step = g.manager.get_step(step_name)
    if step is None:
        return json_error("Step name invalid")

    containers = []
    container_ids = g.manager.containers()
    for container_id in container_ids:
        container = step.get_container(container_id)
        if container is not None:
            containers.append(container)
    return json_response([c.name for c in containers])


@api_blueprint.route("/data/<step_name>/<container_name>")
@login_required
def get_targets(step_name: str, container_name: str):
    try:
        targets = g.manager.get_targets(step_name, container_name)
        return json_response([t.as_dict() for t in targets])
    except RevngException as e:
        return json_error(repr(e))


@api_blueprint.get("/fetch/<step_name>/<container_name>")
@login_required
def fetch_container(step_name, container_name):
    """Fetches the content of the given container"""
    # TODO: fix path-traversal
    logging.warning("Reminder: revng C API is vulnerable to path traversal. Fix it!")
    _container_path = g.manager.container_path(step_name, container_name)
    if _container_path is None:
        return json_error("Invalid step or container name")
    container_path = Path(_container_path)

    if not container_path.is_file():
        return json_error("Cannot fetch the given container (was it produced?)")

    return send_file(container_path, as_attachment=True, etag=True)


@api_blueprint.post("/set-input/<container_name>")
@login_required
def set_input(container_name):
    """Sets the content of the given container"""
    file = request.files.get("input")
    if file is None:
        return json_error("You must provide an input file", http_code=400)

    try:
        container_path = g.manager.set_input(container_name, file.read())
        logging.info(
            f"User {g.user.username} file for container "
            + f"{container_name} saved to {container_path}"
        )

        return json_response({})
    except RevngException as e:
        return json_error(repr(e))


@api_blueprint.get("/features")
@login_required
def features():
    return json_response(
        {
            "kinds": [x.as_dict() for x in g.manager.kinds()],
            "ranks": [x.as_dict() for x in Rank.ranks()],
        }
    )


@api_blueprint.get("/pipeline/describe")
@login_required
def pipeline_describe():
    steps = list(g.manager.steps())

    # TODO: pezzotto code
    last_step = steps[-1]

    containers = []
    for container_id in g.manager.containers():
        containers.append(last_step.get_container(container_id))

    return json_response(
        {"steps": [s.as_dict() for s in steps], "containers": [c.as_dict() for c in containers]}
    )


@api_blueprint.get("/model")
@login_required
def get_model():
    return g.manager.get_model()
