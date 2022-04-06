#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from flask import Blueprint, render_template

demo_blueprint = Blueprint("demo", __name__)


@demo_blueprint.route("/")
def index():
    return render_template("index.html")
