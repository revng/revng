#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import re
from typing import Callable

from revng import ptml
from revng.model import MetaAddress, MetaAddressType  # type: ignore[attr-defined]
from revng.scripting import CLIProfile, DaemonProfile
from revng.scripting.projects import Project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Path to the resume directory", required=True)
    parser.add_argument("--binary", help="Path to the Binary", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cli", help="Run revng cli", action="store_true")
    group.add_argument("--daemon", help="Run revng daemon", action="store_true")
    return parser.parse_args()


def run_test(project_getter: Callable[[str], Project], binary: str, resume: str):
    project = project_getter(resume)
    project.import_and_analyze(binary)

    # Get the first function with `CustomName`
    function_original_name = None
    function_idx = 0
    function_entry = None
    for idx, function in enumerate(project.model.Functions):
        if function.CustomName != "":
            function_original_name = function.CustomName
            function_idx = idx
            function_entry = function.Entry
            break
    assert function_original_name is not None

    # Run an artefact on `TypeDefinitions`
    project.model.TypeDefinitions[1].get_artifact("emit-type-definitions")

    all_functions_entry = [function.Entry for function in project.model.Functions]
    # Assert that when we get the artifact for a single function we get
    # the result of only that function
    result1 = project.model.Functions[function_idx].get_artifact("disassemble")
    parsed1 = ptml.parse(result1)
    re_match = re.search(function_original_name, parsed1)
    assert re_match is not None
    for entry in all_functions_entry:
        if entry == function_entry:
            continue
        re_match = re.search(str(entry), parsed1)
        assert re_match is None

    # Check that when we get the artifact of all the function we get
    # the result for all the functions
    result2 = project.get_artifact("disassemble")
    assert isinstance(result2, bytes)
    parsed2 = ptml.parse(result2)
    for entry in all_functions_entry:
        re_match = re.search(str(entry), parsed2)
        assert re_match is not None

    # Change the function name, get an artifact and make sure that the
    # parsed result contains the new name
    function_new_name1 = "lol"
    project.model.Functions[function_idx].CustomName = function_new_name1
    project.commit()
    result1 = project.model.Functions[function_idx].get_artifact("disassemble")
    assert isinstance(result1, bytes)
    re_match = re.search(function_new_name1, ptml.parse(result1))
    assert re_match is not None

    # Change function name again and commit
    function_new_name2 = "wow"
    project.model.Functions[function_idx].CustomName = function_new_name2
    project.commit()

    # Test revert
    function_new_name3 = "heh"
    project.model.Functions[function_idx].CustomName = function_new_name3
    project.revert()
    assert project.model.Functions[function_idx].CustomName == function_new_name2

    # Stop daemon if `DaemonProject` otherwise the daemon will
    # not persist the data on disk
    if hasattr(project, "stop_daemon"):
        project.stop_daemon()

    # Load the project again and run the analysis again
    project = project_getter(resume)

    project.import_and_analyze(args.binary)

    # make sure that the second name changes was commit
    assert project.model.Functions[function_idx].CustomName == function_new_name2

    # Get the function by passing a MetaAddress
    function = str(project.model.Functions[0].Entry).split(":")
    address = MetaAddress(int(function[0], 16), MetaAddressType[function[1]])

    # Get multiple artifacts
    result = project.get_artifacts(
        {
            "disassemble": [project.model.Functions[address], project.model.Functions[1]],
            "decompile": [],
        }
    )
    assert isinstance(result["disassemble"], bytes)
    assert isinstance(result["decompile"], bytes)
    ptml.parse(result["disassemble"])
    ptml.parse(result["decompile"])

    # Run a few analyses
    for analysis_name in ["detect-stack-size", "analyze-data-layout"]:
        targets = project.get_targets(analysis_name, project.model.Functions)
        project.analyze(analysis_name, targets)


if __name__ == "__main__":
    args = main()

    profile: CLIProfile | DaemonProfile

    if args.cli:
        profile = CLIProfile()
    elif args.daemon:
        profile = DaemonProfile()
    else:
        raise ValueError("The script expects either --cli or --daemon")

    run_test(profile.get_project, args.binary, args.resume)
