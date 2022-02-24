#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from flask import Blueprint, render_template

from .api import login_required

demo_blueprint = Blueprint("demo", __name__)


@demo_blueprint.route("/")
@login_required
def index():
    return render_template("index.html")
