#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import functools
import shutil
import sys
import tarfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from revng.pypeline.cli.context import ContextObject
from revng.pypeline.main import pype

ROOT = Path(__file__).resolve().parent

ENV_FILES = ["pipeline.yml", "pipebox.py", "model.yml"]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(params=["yaml", "tar"])
def container_format(request):
    return request.param


def setup_env(test_func):
    """Decorator to setup an isolated filesystem for CLI tests.
    This won't pass the arguments to the pipebox."""

    @functools.wraps(test_func)
    def wrapped(runner: CliRunner, *args, **kwargs):
        # This is just a chdir to a tmp folder
        with runner.isolated_filesystem():
            # Create the necessary files inside the new filesystem
            for file in ENV_FILES:
                shutil.copyfile(ROOT / file, file)
            # Run the test
            test_func(runner, *args, **kwargs)

    return wrapped


def run_partial(runner: CliRunner, **kwargs):
    """Convenience partial function to pre-fill common args"""

    def run(*args: str):
        result = runner.invoke(cli=pype, args=args, **kwargs)
        if result.exit_code != 0:
            print(result.output, file=sys.stderr)
        assert result.exit_code == 0
        return result

    return run


def is_tar_file(path: Path) -> bool:
    return tarfile.is_tarfile(path)


def is_yaml_file(path: Path) -> bool:
    try:
        with open(path, "r") as f:
            yaml.safe_load(f)
        return True
    except (yaml.YAMLError, FileNotFoundError):
        return False


def check_file_format(path: Path, expected_format: str):
    if expected_format == "yaml":
        assert is_yaml_file(path), f"File {path} is not a valid yaml file"
    elif expected_format == "tar":
        assert is_tar_file(path), f"File {path} is not a valid tar file"
    else:
        raise ValueError(f"Unknown format: {expected_format}")


@setup_env
def test_pipeline(runner: CliRunner, container_format: str):
    run = run_partial(
        runner,
        obj=ContextObject.make(pipebox_args=["first pipebox arg", "second pipebox arg"]),
    )

    # First call some pipes to manually generate some containers
    run(
        "pipeline", "run-pipe", "GeneratorPipe", f"--format={container_format}", "model.yml", "out1"
    )
    check_file_format(Path("out1"), container_format)

    run(
        "pipeline",
        "run-pipe",
        "InPlacePipe",
        f"--format={container_format}",
        "model.yml",
        "out1",
        "out2",
    )
    check_file_format(Path("out2"), container_format)

    run(
        "pipeline",
        "run-pipe",
        "ToLowerKindPipe",
        f"--format={container_format}",
        "model.yml",
        "out2",
        "out3",
    )
    check_file_format(Path("out3"), container_format)

    # Call a couple of analyses
    run(
        "pipeline",
        "run-analysis",
        "NullAnalysis",
        "model.yml",
        "out1",
    )

    run(
        "pipeline",
        "run-analysis",
        "PurgeAllAnalysis",
        "model.yml",
        "out1",
    )

    run(
        "pipeline",
        "run-analysis",
        "AddStuffAnalysis",
        "model.yml",
        "out1",
    )


@setup_env
def test_project(runner: CliRunner, container_format: str):
    run = run_partial(
        runner,
        obj=ContextObject.make(pipebox_args=["first pipebox arg", "second pipebox arg"]),
    )

    # Compute artifacts
    run(
        "project",
        "artifact",
        "ChildArtifact",
        f"--format={container_format}",
        "-o",
        "child_artifact",
    )
    check_file_format(Path("child_artifact"), container_format)

    run(
        "project",
        "artifact",
        "RootArtifact",
        f"--format={container_format}",
        "-o",
        "root_artifact",
    )
    check_file_format(Path("root_artifact"), container_format)

    # Run analysis
    run("project", "analyze", "NullAnalysis")
    run("project", "analyze", "PurgeAllAnalysis")
    run("project", "analyze", "AddStuffAnalysis")

    # Run an analysis list
    run("project", "analyze", "all_analyses")
