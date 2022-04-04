#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from pytest import Parser, hookimpl


def pytest_addoption(parser: Parser):
    parser.addoption("--binary", action="store")
    parser.addoption("--root", action="store")


@hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)
