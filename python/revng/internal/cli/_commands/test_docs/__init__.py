#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import abc
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

import marko

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options

SCRIPT_DIR = Path(__file__).parent.resolve()
verbose = False


def log(message):
    if verbose:
        sys.stderr.write(message + "\n")


def run(working_directory: Path, arguments):
    log(f"Running {shlex.join(arguments)}")
    process = subprocess.run(
        arguments, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    if process.returncode != 0:
        sys.stderr.write(f"The following program failed: {shlex.join(arguments)}\n")
        sys.stderr.flush()
        sys.stderr.buffer.write(process.stdout)
        sys.stderr.flush()
        sys.stderr.write(f"Process exited with code {process.returncode}\n")
        sys.stderr.write("Re-run with --verbose for further info\n")
        sys.exit(1)


class Doctest(abc.ABC):
    @abc.abstractmethod
    def run(self, working_directory: Path):
        pass

    @abc.abstractmethod
    def process(self, code: str, extra: str = ""):
        pass

    @staticmethod
    def _parse_extra(extra: str) -> dict[str, str | None]:
        if extra.strip() == "":
            return {}

        result: dict[str, str | None] = {}
        for part in re.split(r"\s+", extra):
            assert part.count("=") in (0, 1)
            if part.count("=") == 0:
                result[part] = None
            else:
                key, value = part.split("=", 1)
                result[key] = value
        return result


class PythonDoctest(Doctest):
    def __init__(self):
        self.script = ""

    def run(self, working_directory: Path):
        if not self.script:
            return

        self.script = f'"""{self.script}"""'
        log(f"Running the following Python script:\n{textwrap.indent(self.script, '    ')}\n")

        script_path = working_directory / "run.py"
        script_path.write_text(self.script)
        run(working_directory, ["python", str(SCRIPT_DIR / "doctest_runner.py"), "-v", "run.py"])
        self.script = ""

    def process(self, code: str, extra: str = ""):
        extra_decoded = self._parse_extra(extra)
        if "ignoreoutput" in extra_decoded:
            ignore_output = extra_decoded["ignoreoutput"]
            split_lines = code.splitlines()

            if ignore_output is None:
                lines_to_ignore_output = list(range(1, len(split_lines) + 1))
            else:
                lines_to_ignore_output = [int(x) for x in ignore_output.split(",")]

            for index, line in enumerate(split_lines):
                if index + 1 in lines_to_ignore_output:
                    self.script += line + "  # doctest: +IGNORE_OUTPUT\n"
                else:
                    self.script += line + "\n"
            self.script += "\n"
        else:
            self.script += code + "\n"


class BashDoctest(Doctest):
    def __init__(self):
        self.script = ""
        self.expected_output = ""

    def run(self, working_directory: Path):
        if not self.script and not self.expected_output:
            return

        self.script = self.script.strip() + "\n"
        script_path = working_directory / "run.sh"
        script_path.write_text(self.script)

        self.expected_output = self.expected_output.strip() + "\n"
        expected_output_path = working_directory / "expected_output.log"
        expected_output_path.write_text(self.expected_output)

        log(f"Running the following bash script:\n{textwrap.indent(self.script, '    ')}\n")
        log(f"Expected output is:\n{textwrap.indent(self.expected_output, '    ')}\n")

        run(
            working_directory,
            [
                "bash",
                "-c",
                "bash -euo pipefail ./run.sh |& tee output.log"
                + " && diff -Bwu output.log expected_output.log",
            ],
        )
        self.script = ""
        self.expected_output = ""

    def process(self, code, extra=""):
        next_line_is_command = False
        silent = "silent" in extra
        match = re.match('.*ignore="([^"]*)".*', extra)
        ignore_regexp = None
        if match:
            ignore_regexp = match.groups()[0]
            self.script += "( "

        if silent:
            self.script += "( "

        for line in code.split("\n"):
            if line.startswith("$ "):
                self.script += line[2:] + "\n"
                if line.endswith("\\"):
                    next_line_is_command = True
            elif next_line_is_command:
                self.script += line + "\n"
                next_line_is_command = line.endswith("\\")
            else:
                if not ignore_regexp or not re.match(".*(" + ignore_regexp + ").*", line):
                    self.expected_output += line + "\n"

        if silent:
            self.script += " ) >& /dev/null\n"

        if ignore_regexp:
            self.script += f" ) |& grep -vE '{ignore_regexp}'\n"


class TypeScriptDoctest(Doctest):
    def __init__(self):
        self.script = ""

    def run(self, working_directory: Path):
        if not self.script:
            return

        self.script = self.script.strip() + "\n"
        log(f"Running the following TypeScript script:\n{textwrap.indent(self.script, '    ')}")
        script_path = working_directory / "run.ts"
        script_path.write_text(self.script)
        run(
            working_directory,
            [
                "bash",
                "-c",
                "npm install typescript revng-model @types/node"
                + " && ./node_modules/.bin/tsc run.ts"
                + " && node run.js",
            ],
        )
        self.script = ""

    @staticmethod
    def escape_js(string):
        assert type(string) is str
        return json.dumps(string)

    @classmethod
    def emit_assertion(cls, expected):
        return f"if (JSON.stringify(last) !== {cls.escape_js(expected[:-1])})\n  process.exit(1);\n"

    def process(self, code, extra=""):
        output = "console.log = (x) => { return x; };\nlet last;\n"

        last_output = ""
        lines = list(map(str.strip, code.split("\n")))[:-1]
        for line, next_line in zip(lines, lines[1:] + ["> "]):
            is_command = line.startswith("> ")
            next_is_command = next_line.startswith("> ")

            if is_command:
                if last_output:
                    output += self.emit_assertion(last_output)
                    last_output = ""

                line = line[2:]
                if next_is_command:
                    output += f"{line}\n"
                else:
                    output += f"last = {line}\n"

            else:
                last_output += line + "\n"

        if last_output:
            output += self.emit_assertion(last_output)

        self.script = output


def only(entries):
    assert len(entries) == 1
    return entries[0]


def handle_file(path: Path):
    log(f"Processing {str(path)}")

    handler_types = {
        "python": PythonDoctest,
        "bash": BashDoctest,
        "typescript": TypeScriptDoctest,
    }

    handlers: Mapping[str, Doctest] = {
        language: constructor() for language, constructor in handler_types.items()  # type: ignore
    }

    document = marko.parse(path.read_text())

    for block in document.children:
        if type(block) is not marko.block.FencedCode:
            continue

        block.lang = block.lang.strip("{").strip("}")
        block.extra = block.extra.strip("{").strip("}")

        if "notest" in block.extra:
            log(
                f"Ignoring fenced code of type {block.lang}"
                + f' due to `notest` in "{block.extra}"'
            )
            continue

        if "noorchestra" in block.extra:
            log(f'Disabling orchestra environment due to `noorchestra` in "{block.extra}"')
            handlers["bash"].process(
                "$ unset ORCHESTRA_DOTDIR;"
                + " unset ORCHESTRA_ROOT;"
                + " unset ORCHESTRA_NODE_CACHE;"
                + " export ORCHESTRA_DOTDIR ORCHESTRA_ROOT ORCHESTRA_NODE_CACHE;"
            )

        rawtext = only(block.children)
        assert type(rawtext) is marko.inline.RawText
        text = rawtext.children
        assert type(text) is str

        match = re.match(r"title=([^ ]*\.[^ ]*)", block.extra)
        if match:
            append_to = match.groups()[0]
            log(f"Appending to {append_to}")
            if text.endswith("\n"):
                text = text[:-1]
            for line in text.split("\n"):
                handlers["bash"].process("$ echo " + shlex.quote(line) + " >> " + append_to)
        elif block.lang == "diff":
            log("Applying diff")
            handlers["bash"].process("""$ rm -f patch.patch""")
            for line in text.split("\n"):
                handlers["bash"].process("$ echo " + shlex.quote(line) + " >> patch.patch")
            handlers["bash"].process("""$ patch --quiet -p1 < patch.patch""")
            handlers["bash"].process("""$ rm patch.patch""")

        elif block.lang in handlers:
            log(f"Handling {block.lang} snippet")
            handlers[block.lang].process(text, block.extra)

    for language, handler in handlers.items():
        with TemporaryDirectory(
            prefix=f"revng-docs-test-{language}-{os.path.basename(path)}-"
        ) as temporary_directory:
            handler.run(Path(temporary_directory))


class TestDocsCommand(Command):
    def __init__(self):
        super().__init__(("test-docs",), "Test mkdocs files")

    def register_arguments(self, parser):
        parser.add_argument("files", metavar="FILE", nargs="+", help="files to test.")
        parser.add_argument("--verbose", action="store_true", help="enable verbose output.")

    def run(self, options: Options):
        args = options.parsed_args

        global verbose
        verbose = args.verbose

        for path in args.files:
            handle_file(Path(path))


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(TestDocsCommand())
