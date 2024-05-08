#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#
from pytest import Parser, TestReport, hookimpl


def pytest_addoption(parser: Parser):
    parser.addoption("--binary", action="store")


@hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    rep: TestReport = (yield).get_result()
    setattr(item, "rep_" + rep.when, rep)
