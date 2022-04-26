#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
from typing import TYPE_CHECKING, cast

from flask import Blueprint, current_app, jsonify, request

from ariadne import graphql
from ariadne.constants import PLAYGROUND_HTML
from ariadne.contrib.tracing.apollotracing import ApolloTracingExtension
from ariadne.file_uploads import FilesDict, combine_multipart_data

from .schema import SchemafulManager

if TYPE_CHECKING:
    from flask.ctx import _AppCtxGlobals

    class FlaskGlobals(_AppCtxGlobals):
        manager: SchemafulManager

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


# When navigating to http://<server>/graphql show a graphql sandbox
@api_blueprint.route("/graphql", methods=["GET"])
def graphql_playground():
    return PLAYGROUND_HTML, 200


@api_blueprint.route("/graphql", methods=["POST"])
async def graphql_server():
    if request.content_type.startswith("multipart/form-data;"):
        operations = request.form.get("operations")
        req_map = request.form.get("map")
        if operations is not None and req_map is not None:
            request_files = cast(FilesDict, request.files)
            data = combine_multipart_data(
                json.loads(operations), json.loads(req_map), request_files
            )
        else:
            return json_error("Invalid form data")
    else:
        data = request.get_json()

    success, result = await graphql(
        g.manager.schema,
        data,
        context_value={"g": g, "request": request},
        extensions=[ApolloTracingExtension],
        debug=current_app.config["DEBUG"],
    )

    status_code = 200 if success else 400
    return jsonify(result), status_code
