#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import re
from typing import Callable, Dict, List, Mapping

from revng.project import CLIProject, LocalDaemonProject, Project
from revng.support.artifacts import PTMLArtifact


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", help="Path to the project directory", required=True)
    parser.add_argument("--binary", help="Path to the Binary", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cli", help="Run revng cli", action="store_true")
    group.add_argument("--daemon", help="Run revng daemon", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.cli:

        def project_getter():
            return CLIProject(args.project_dir)

    elif args.daemon:

        def project_getter():
            return LocalDaemonProject(args.project_dir, connection_retries=100)

    else:
        raise ValueError("The script expects either --cli or --daemon")

    run_test(project_getter, args.binary)


def run_test(project_getter: Callable[[], Project], binary: str):
    project = project_getter()
    project.import_and_analyze(binary)

    # Get the first function with `Name`
    function_original_name = None
    function_idx = 0
    function_entry = None
    for idx, function in enumerate(project.model.Functions):
        if function.Name != "":
            function_original_name = function.Name
            function_idx = idx
            function_entry = function.Entry
            break
    assert function_original_name is not None

    # Run an artefact on `TypeDefinitions`
    project.model.TypeDefinitions[1].get_artifact("emit-type-definitions")

    all_functions_entries = [function.Entry for function in project.model.Functions]
    # Assert that when we get the artifact for a single function we get
    # the result of only that function
    the_function = project.model.Functions[function_idx]
    result1 = the_function.get_artifact("disassemble")
    result1_accessor = the_function.disassemble
    assert isinstance(result1, PTMLArtifact)
    assert isinstance(result1_accessor, PTMLArtifact)
    assert result1.parse().text == result1_accessor.parse().text
    parsed1 = result1.parse()
    re_match = re.search(function_original_name, parsed1.text)
    assert re_match is not None
    for entry in all_functions_entries:
        if entry == function_entry:
            continue
        re_match = re.search(str(entry), parsed1.text)
        assert re_match is None

    # Check that when we get the artifact of all the function we get
    # the result for all the functions
    result2 = project.model.get_artifact("disassemble")
    assert isinstance(result2, Mapping)
    all_functions_locations = {function._location for function in project.model.Functions}
    assert set(result2.keys()) == all_functions_locations
    for value in result2.values():
        assert isinstance(value, PTMLArtifact)
        value.parse()

    # Change the function name, get an artifact and make sure that the
    # parsed result contains the new name
    function_new_name1 = "lol"
    project.model.Functions[function_idx].Name = function_new_name1
    project.model.commit()
    result3 = project.model.Functions[function_idx].get_artifact("disassemble")
    assert isinstance(result3, PTMLArtifact)
    parsed3 = result3.parse()
    re_match = re.search(function_new_name1, parsed3.text)
    assert re_match is not None

    # Change function name again and commit
    function_new_name2 = "wow"
    project.model.Functions[function_idx].Name = function_new_name2
    project.model.commit()

    # Test revert
    function_new_name3 = "heh"
    project.model.Functions[function_idx].Name = function_new_name3
    project.model.revert()
    assert project.model.Functions[function_idx].Name == function_new_name2

    # Stop daemon if `DaemonProject` otherwise the daemon will
    # not persist the data on disk
    if isinstance(project, LocalDaemonProject):
        project._stop_daemon()

    # Load the project again, this should also load the model and
    # pipeline description
    project = project_getter()
    project.upload_binary(binary)

    # make sure that the second name changes was commit
    assert project.model.Functions[function_idx].Name == function_new_name2

    # Run an analysis with objects
    analysis_name = "detect-stack-size"
    containers: Dict[str, List[str]] = {
        "llvm-functions": [f._location for f in project.model.Functions]
    }
    project.model.analyze(analysis_name, containers=containers)

    # Run an analysis without objects
    project.model.analyze(analysis_name)


if __name__ == "__main__":
    main()
