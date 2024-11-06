#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from abc import ABC
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

import marko

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options

verbose = False


def log(message):
    global verbose
    if verbose:
        sys.stderr.write(message + "\n")


def run(working_directory, arguments):
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


class Doctest(ABC):
    def run(self, working_directory):
        pass

    def process(self, code, extra=""):
        pass


class PythonDoctest(Doctest):
    def __init__(self):
        self.script = ""

    def run(self, working_directory):
        if not self.script:
            return

        self.script = f"\"\"\"{self.script}\"\"\""
        log(f"Running the following Python script:\n{textwrap.indent(self.script, '    ')}\n")

        script_path = working_directory / "run.py"
        script_path.write_text(self.script)
        run(
            working_directory,
            ["python", "-m", "doctest", "-v", "run.py"]
        )
        self.script = ""

    def process(self, code, extra=""):
        self.script += code + "\n"


class BashDoctest(Doctest):
    def __init__(self):
        self.script = ""
        self.expected_output = ""

    def run(self, working_directory):
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


def escape_js(string):
    assert type(string) is str
    return json.dumps(string)


def emit_assertion(expected):
    return f"if (JSON.stringify(last) !== {escape_js(expected[:-1])})\n  process.exit(1);\n"


class TypeScriptDoctest(Doctest):
    def __init__(self):
        self.script = ""

    def run(self, working_directory):
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
                + " && node run.js"
            ],
        )
        self.script = ""

    def process(self, code, extra=""):
        output = "console.log = (x) => { return x; };\nlet last;\n"

        last_output = ""
        lines = list(map(str.strip, code.split("\n")))[:-1]
        for line, next_line in zip(lines, lines[1:] + ["> "]):
            is_command = line.startswith("> ")
            next_is_command = next_line.startswith("> ")

            if is_command:
                if last_output:
                    output += emit_assertion(last_output)
                    last_output = ""

                line = line[2:]
                if next_is_command:
                    output += f"{line}\n"
                else:
                    output += f"last = {line}\n"

            else:
                last_output += line + "\n"

        if last_output:
            output += emit_assertion(last_output)

        self.script = output


def only(entries):
    assert len(entries) == 1
    return entries[0]


def handle_file(path: Path):
    handler_types = {
        "python": PythonDoctest,
        "bash": BashDoctest,
        "typescript": TypeScriptDoctest,
    }

    handlers: Mapping[str, Doctest] = {
        language: constructor() for language, constructor in handler_types.items()
    }

    document = marko.parse(path.read_text())

    with TemporaryDirectory(
        prefix=f"revng-docs-test-{os.path.basename(path)}-"
    ) as temporary_directory:
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
                handlers[block.lang].run(Path(temporary_directory))


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
