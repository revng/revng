#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import logging
from pathlib import Path

from flask import Blueprint, g, jsonify, request, send_file

from revng.api.target import Target
from revng.api.rank import Rank
from .auth import login_required
from .validation_utils import all_strings

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


@api_blueprint.route("/steps")
@login_required
def steps():
    """Returns the list of the existing step names"""
    return json_response([s.name for s in g.manager.steps()])


@api_blueprint.route("/containers")
@login_required
def containers():
    """Returns the list of the existing container names"""
    return json_response([c.name for c in g.manager.containers()])


@api_blueprint.post("/containers/store_all")
@login_required
def store_all_containers():
    """Stores all containers to disk"""
    # TODO: is this endpoint needed?
    success = g.manager.store_containers()
    if success:
        return json_response()
    else:
        return json_error("Failed", http_code=500)


@api_blueprint.post("/produce")
@login_required
def produce():
    # TODO: does it make sense to allow producing a specific target? Wouldn't it
    #       make sense to allow producing only a given container?
    step_name = request.json["step"]
    container_name = request.json["container"]
    kind_name = request.json["kind"]
    exact = bool(request.json.get("exact", True))
    path_components = request.json["path_components"]

    step = g.manager.get_step(step_name)
    if step is None:
        return json_error("Invalid step")

    container_identifier = g.manager.get_container_with_name(container_name)
    if container_identifier is None:
        return json_error("Invalid container")

    container = step.get_container(container_identifier)
    if container is None:
        return json_error(f"Step {step_name} does not use container {container_name}")

    kind = g.manager.kind_from_name(kind_name)
    if kind is None:
        return json_error("Invalid kind")

    if not isinstance(path_components, list) or not all_strings(path_components):
        return json_error("Invalid path components")

    target = Target.create(kind, exact, path_components)
    if target is None:
        return json_error("Invalid target")

    produce_success = g.manager.produce_target(target, step, container)
    if not produce_success:
        # TODO: we really should be able to provide a detailed error here
        return json_error("Failed to produce target", http_code=500)

    # TODO: is it ok to store the produced target here? Should we use Container.store?
    #       If so, can we reuse the previous container or do we need to make a new instance?
    store_success = g.manager.store_containers()
    if not store_success:
        return json_error("Target produced successfully but not stored", http_code=500)

    return json_response({})


@api_blueprint.route("/step/<step_name>/<container_name>/targets")
@login_required
def step_targets(step_name, container_name):
    """Returns the list of targets for the given container/step tuple"""
    step = g.manager.get_step(step_name)
    if step is None:
        return json_error("Invalid step name")

    container_identifier = g.manager.get_container_with_name(container_name)
    if container_identifier is None:
        return json_error("Invalid container name")

    container = step.get_container(container_identifier)
    if container is None:
        return json_error(f"Step {step_name} does not use container {container_name}")

    targets_list = g.manager.get_targets_list(container)
    if targets_list is None:
        return json_error("Invalid container name (cannot get targets list)")

    return json_response([t.as_dict() for t in targets_list.targets()])


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


@api_blueprint.post("/set/<step_name>/<container_name>")
@login_required
def set_input(step_name, container_name):
    """Sets the content of the given container"""
    # TODO: do we want to allow to set any container or just Lift/input?
    step = g.manager.get_step(step_name)
    if step is None:
        return json_error("Invalid step name")

    container_identifier = g.manager.get_container_with_name(container_name)
    if container_identifier is None:
        return json_error("Invalid container name")

    container = step.get_container(container_identifier)
    if container is None:
        return json_error(f"Step {step_name} does not use container {container_name}")

    file = request.files.get("input")
    if file is None:
        return json_error("You must provide an input file", http_code=400)

    # TODO: fix path traversal
    logging.warning("Reminder: revng C API is vulnerable to path traversal. Fix it!")
    _container_path = g.manager.container_path(step_name, container_name)
    if _container_path is None:
        return json_error("Invalid step or container name")
    container_path = Path(_container_path)
    container_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(container_path)
    logging.info(
        f"User {g.user.username} file for container "
        + f"{step_name}/{container_name} saved to {container_path}"
    )

    success = container.load(str(container_path))
    if not success:
        return json_error(f"Failed loading user provided input for container {container_name}")

    return json_response({})


@api_blueprint.get("/features")
@login_required
def features():
    return json_response(
        {
            "kinds": [x.as_dict() for x in g.manager.kinds()],
            "ranks": [x.as_dict() for x in Rank.ranks()],
        }
    )
