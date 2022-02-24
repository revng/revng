#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import logging
import tempfile
from functools import wraps

from flask import Blueprint, g, redirect, render_template, request, session, url_for

from .globals import managers

auth_blueprint = Blueprint("auth", __name__)


class User:
    def __init__(self, username, tmpdir):
        self.username = username
        self.tmpdir = tmpdir

    @staticmethod
    def from_dict(d: dict):
        return User(d["username"], d["tmpdir"])


def get_logged_user():
    user_dict = session.get("user")
    if not user_dict:
        return None
    return User.from_dict(user_dict)


def is_logged_in():
    return session.get("user") is not None


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_logged_in():
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)

    return decorated_function


@auth_blueprint.route("/login", methods=["GET", "POST"])
def login():
    # TODO: actually implement proper authentication and session management
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username")
    if not username:
        raise ValueError("Provide a username")

    session["user"] = {"username": username, "tmpdir": tempfile.mkdtemp()}

    return redirect(url_for("demo.index"))


@auth_blueprint.route("/logout")
def logout():
    # TODO: we should delete temporary files, free managers, etc
    if g.user and g.user.username in managers:
        logging.info(f"Releasing manager for {g.user.username}")
        del managers[g.user.username]

    session.clear()
    return redirect(url_for("auth.login"))
